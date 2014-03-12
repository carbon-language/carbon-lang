//===--- CodeGenPGO.cpp - PGO Instrumentation for LLVM CodeGen --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Instrumentation-based profile-guided optimization
//
//===----------------------------------------------------------------------===//

#include "CodeGenPGO.h"
#include "CodeGenFunction.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "llvm/Config/config.h" // for strtoull()/strtoll() define
#include "llvm/IR/MDBuilder.h"
#include "llvm/Profile/ProfileDataReader.h"
#include "llvm/Support/FileSystem.h"

using namespace clang;
using namespace CodeGen;

void CodeGenPGO::setFuncName(llvm::Function *Fn) {
  StringRef Func = Fn->getName();

  // Function names may be prefixed with a binary '1' to indicate
  // that the backend should not modify the symbols due to any platform
  // naming convention. Do not include that '1' in the PGO profile name.
  if (Func[0] == '\1')
    Func = Func.substr(1);

  if (!Fn->hasLocalLinkage()) {
    FuncName = new std::string(Func);
    return;
  }

  // For local symbols, prepend the main file name to distinguish them.
  // Do not include the full path in the file name since there's no guarantee
  // that it will stay the same, e.g., if the files are checked out from
  // version control in different locations.
  FuncName = new std::string(CGM.getCodeGenOpts().MainFileName);
  if (FuncName->empty())
    FuncName->assign("<unknown>");
  FuncName->append(":");
  FuncName->append(Func);
}

void CodeGenPGO::emitWriteoutFunction() {
  if (!CGM.getCodeGenOpts().ProfileInstrGenerate)
    return;

  llvm::LLVMContext &Ctx = CGM.getLLVMContext();

  llvm::Type *Int32Ty = llvm::Type::getInt32Ty(Ctx);
  llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(Ctx);

  llvm::Function *AddFuncsF =
    CGM.getModule().getFunction("__llvm_pgo_add_functions");
  if (!AddFuncsF) {
    llvm::FunctionType *AddFuncsFTy =
      llvm::FunctionType::get(llvm::Type::getVoidTy(Ctx), false);
    AddFuncsF = llvm::Function::Create(AddFuncsFTy,
                                       llvm::GlobalValue::InternalLinkage,
                                       "__llvm_pgo_add_functions",
                                       &CGM.getModule());
  }
  AddFuncsF->setUnnamedAddr(true);
  AddFuncsF->addFnAttr(llvm::Attribute::NoInline);
  if (CGM.getCodeGenOpts().DisableRedZone)
    AddFuncsF->addFnAttr(llvm::Attribute::NoRedZone);

  llvm::BasicBlock *BB = AddFuncsF->empty() ?
    llvm::BasicBlock::Create(Ctx, "", AddFuncsF) : &AddFuncsF->getEntryBlock();

  CGBuilderTy PGOBuilder(BB);

  llvm::Instruction *I = BB->getTerminator();
  if (!I)
    I = PGOBuilder.CreateRetVoid();
  PGOBuilder.SetInsertPoint(I);

  llvm::Type *Int64PtrTy = llvm::Type::getInt64PtrTy(Ctx);
  llvm::Type *Int64Ty = llvm::Type::getInt64Ty(Ctx);
  llvm::Type *Args[] = {
    Int8PtrTy,                       // const char *FuncName
    Int64Ty,                         // uint64_t FunctionHash
    Int32Ty,                         // uint32_t NumCounters
    Int64PtrTy                       // uint64_t *Counters
  };
  llvm::FunctionType *FTy =
    llvm::FunctionType::get(PGOBuilder.getVoidTy(), Args, false);
  llvm::Constant *EmitFunc =
    CGM.getModule().getOrInsertFunction("llvm_pgo_add_function", FTy);

  llvm::Constant *NameString =
    CGM.GetAddrOfConstantCString(getFuncName(), "__llvm_pgo_name");
  NameString = llvm::ConstantExpr::getBitCast(NameString, Int8PtrTy);
  PGOBuilder.CreateCall4(EmitFunc, NameString,
                         // TODO: This should be a hash, not just the count!
                         PGOBuilder.getInt64(NumRegionCounters),
                         PGOBuilder.getInt32(NumRegionCounters),
                         PGOBuilder.CreateBitCast(RegionCounters, Int64PtrTy));
}

llvm::Function *CodeGenPGO::emitInitialization(CodeGenModule &CGM) {
  llvm::Function *AddFuncsF =
    CGM.getModule().getFunction("__llvm_pgo_add_functions");
  if (!AddFuncsF)
    return NULL;

  // Create a small bit of code that initializes PGO at startup
  llvm::Function *F = CGM.getModule().getFunction("__llvm_pgo_init");
  if (F)
    return NULL;

  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  llvm::FunctionType *FTy = llvm::FunctionType::get(llvm::Type::getVoidTy(Ctx),
                                                    false);
  F = llvm::Function::Create(FTy, llvm::GlobalValue::InternalLinkage,
                             "__llvm_pgo_init", &CGM.getModule());
  F->setUnnamedAddr(true);
  F->setLinkage(llvm::GlobalValue::InternalLinkage);
  F->addFnAttr(llvm::Attribute::NoInline);
  if (CGM.getCodeGenOpts().DisableRedZone)
    F->addFnAttr(llvm::Attribute::NoRedZone);

  llvm::BasicBlock *BB = llvm::BasicBlock::Create(CGM.getLLVMContext(), "", F);
  CGBuilderTy PGOBuilder(BB);

  FTy = llvm::FunctionType::get(PGOBuilder.getVoidTy(), false);
  llvm::Type *Params[] = {
    llvm::PointerType::get(FTy, 0)
  };
  FTy = llvm::FunctionType::get(PGOBuilder.getVoidTy(), Params, false);

  // Inialize the environment and register the local writeout function.
  llvm::Constant *PGOInit =
    CGM.getModule().getOrInsertFunction("llvm_pgo_init", FTy);
  PGOBuilder.CreateCall(PGOInit, AddFuncsF);
  PGOBuilder.CreateRetVoid();

  return F;
}

namespace {
  /// A StmtVisitor that fills a map of statements to PGO counters.
  struct MapRegionCounters : public ConstStmtVisitor<MapRegionCounters> {
    /// The next counter value to assign.
    unsigned NextCounter;
    /// The map of statements to counters.
    llvm::DenseMap<const Stmt*, unsigned> *CounterMap;

    MapRegionCounters(llvm::DenseMap<const Stmt*, unsigned> *CounterMap) :
      NextCounter(0), CounterMap(CounterMap) {
    }

    void VisitChildren(const Stmt *S) {
      for (Stmt::const_child_range I = S->children(); I; ++I)
        if (*I)
         this->Visit(*I);
    }
    void VisitStmt(const Stmt *S) { VisitChildren(S); }

    /// Assign a counter to track entry to the function body.
    void VisitFunctionDecl(const FunctionDecl *S) {
      (*CounterMap)[S->getBody()] = NextCounter++;
      Visit(S->getBody());
    }
    void VisitObjCMethodDecl(const ObjCMethodDecl *S) {
      (*CounterMap)[S->getBody()] = NextCounter++;
      Visit(S->getBody());
    }
    void VisitBlockDecl(const BlockDecl *S) {
      (*CounterMap)[S->getBody()] = NextCounter++;
      Visit(S->getBody());
    }
    /// Assign a counter to track the block following a label.
    void VisitLabelStmt(const LabelStmt *S) {
      (*CounterMap)[S] = NextCounter++;
      Visit(S->getSubStmt());
    }
    /// Assign a counter for the body of a while loop.
    void VisitWhileStmt(const WhileStmt *S) {
      (*CounterMap)[S] = NextCounter++;
      Visit(S->getCond());
      Visit(S->getBody());
    }
    /// Assign a counter for the body of a do-while loop.
    void VisitDoStmt(const DoStmt *S) {
      (*CounterMap)[S] = NextCounter++;
      Visit(S->getBody());
      Visit(S->getCond());
    }
    /// Assign a counter for the body of a for loop.
    void VisitForStmt(const ForStmt *S) {
      (*CounterMap)[S] = NextCounter++;
      if (S->getInit())
        Visit(S->getInit());
      const Expr *E;
      if ((E = S->getCond()))
        Visit(E);
      if ((E = S->getInc()))
        Visit(E);
      Visit(S->getBody());
    }
    /// Assign a counter for the body of a for-range loop.
    void VisitCXXForRangeStmt(const CXXForRangeStmt *S) {
      (*CounterMap)[S] = NextCounter++;
      Visit(S->getRangeStmt());
      Visit(S->getBeginEndStmt());
      Visit(S->getCond());
      Visit(S->getLoopVarStmt());
      Visit(S->getBody());
      Visit(S->getInc());
    }
    /// Assign a counter for the body of a for-collection loop.
    void VisitObjCForCollectionStmt(const ObjCForCollectionStmt *S) {
      (*CounterMap)[S] = NextCounter++;
      Visit(S->getElement());
      Visit(S->getBody());
    }
    /// Assign a counter for the exit block of the switch statement.
    void VisitSwitchStmt(const SwitchStmt *S) {
      (*CounterMap)[S] = NextCounter++;
      Visit(S->getCond());
      Visit(S->getBody());
    }
    /// Assign a counter for a particular case in a switch. This counts jumps
    /// from the switch header as well as fallthrough from the case before this
    /// one.
    void VisitCaseStmt(const CaseStmt *S) {
      (*CounterMap)[S] = NextCounter++;
      Visit(S->getSubStmt());
    }
    /// Assign a counter for the default case of a switch statement. The count
    /// is the number of branches from the loop header to the default, and does
    /// not include fallthrough from previous cases. If we have multiple
    /// conditional branch blocks from the switch instruction to the default
    /// block, as with large GNU case ranges, this is the counter for the last
    /// edge in that series, rather than the first.
    void VisitDefaultStmt(const DefaultStmt *S) {
      (*CounterMap)[S] = NextCounter++;
      Visit(S->getSubStmt());
    }
    /// Assign a counter for the "then" part of an if statement. The count for
    /// the "else" part, if it exists, will be calculated from this counter.
    void VisitIfStmt(const IfStmt *S) {
      (*CounterMap)[S] = NextCounter++;
      Visit(S->getCond());
      Visit(S->getThen());
      if (S->getElse())
        Visit(S->getElse());
    }
    /// Assign a counter for the continuation block of a C++ try statement.
    void VisitCXXTryStmt(const CXXTryStmt *S) {
      (*CounterMap)[S] = NextCounter++;
      Visit(S->getTryBlock());
      for (unsigned I = 0, E = S->getNumHandlers(); I < E; ++I)
        Visit(S->getHandler(I));
    }
    /// Assign a counter for a catch statement's handler block.
    void VisitCXXCatchStmt(const CXXCatchStmt *S) {
      (*CounterMap)[S] = NextCounter++;
      Visit(S->getHandlerBlock());
    }
    /// Assign a counter for the "true" part of a conditional operator. The
    /// count in the "false" part will be calculated from this counter.
    void VisitConditionalOperator(const ConditionalOperator *E) {
      (*CounterMap)[E] = NextCounter++;
      Visit(E->getCond());
      Visit(E->getTrueExpr());
      Visit(E->getFalseExpr());
    }
    /// Assign a counter for the right hand side of a logical and operator.
    void VisitBinLAnd(const BinaryOperator *E) {
      (*CounterMap)[E] = NextCounter++;
      Visit(E->getLHS());
      Visit(E->getRHS());
    }
    /// Assign a counter for the right hand side of a logical or operator.
    void VisitBinLOr(const BinaryOperator *E) {
      (*CounterMap)[E] = NextCounter++;
      Visit(E->getLHS());
      Visit(E->getRHS());
    }
  };

  /// A StmtVisitor that propagates the raw counts through the AST and
  /// records the count at statements where the value may change.
  struct ComputeRegionCounts : public ConstStmtVisitor<ComputeRegionCounts> {
    /// PGO state.
    CodeGenPGO &PGO;

    /// A flag that is set when the current count should be recorded on the
    /// next statement, such as at the exit of a loop.
    bool RecordNextStmtCount;

    /// The map of statements to count values.
    llvm::DenseMap<const Stmt*, uint64_t> *CountMap;

    /// BreakContinueStack - Keep counts of breaks and continues inside loops. 
    struct BreakContinue {
      uint64_t BreakCount;
      uint64_t ContinueCount;
      BreakContinue() : BreakCount(0), ContinueCount(0) {}
    };
    SmallVector<BreakContinue, 8> BreakContinueStack;

    ComputeRegionCounts(llvm::DenseMap<const Stmt*, uint64_t> *CountMap,
                        CodeGenPGO &PGO) :
      PGO(PGO), RecordNextStmtCount(false), CountMap(CountMap) {
    }

    void RecordStmtCount(const Stmt *S) {
      if (RecordNextStmtCount) {
        (*CountMap)[S] = PGO.getCurrentRegionCount();
        RecordNextStmtCount = false;
      }
    }

    void VisitStmt(const Stmt *S) {
      RecordStmtCount(S);
      for (Stmt::const_child_range I = S->children(); I; ++I) {
        if (*I)
         this->Visit(*I);
      }
    }

    void VisitFunctionDecl(const FunctionDecl *S) {
      RegionCounter Cnt(PGO, S->getBody());
      Cnt.beginRegion();
      (*CountMap)[S->getBody()] = PGO.getCurrentRegionCount();
      Visit(S->getBody());
    }

    void VisitObjCMethodDecl(const ObjCMethodDecl *S) {
      RegionCounter Cnt(PGO, S->getBody());
      Cnt.beginRegion();
      (*CountMap)[S->getBody()] = PGO.getCurrentRegionCount();
      Visit(S->getBody());
    }

    void VisitBlockDecl(const BlockDecl *S) {
      RegionCounter Cnt(PGO, S->getBody());
      Cnt.beginRegion();
      (*CountMap)[S->getBody()] = PGO.getCurrentRegionCount();
      Visit(S->getBody());
    }

    void VisitReturnStmt(const ReturnStmt *S) {
      RecordStmtCount(S);
      if (S->getRetValue())
        Visit(S->getRetValue());
      PGO.setCurrentRegionUnreachable();
      RecordNextStmtCount = true;
    }

    void VisitGotoStmt(const GotoStmt *S) {
      RecordStmtCount(S);
      PGO.setCurrentRegionUnreachable();
      RecordNextStmtCount = true;
    }

    void VisitLabelStmt(const LabelStmt *S) {
      RecordNextStmtCount = false;
      RegionCounter Cnt(PGO, S);
      Cnt.beginRegion();
      (*CountMap)[S] = PGO.getCurrentRegionCount();
      Visit(S->getSubStmt());
    }

    void VisitBreakStmt(const BreakStmt *S) {
      RecordStmtCount(S);
      assert(!BreakContinueStack.empty() && "break not in a loop or switch!");
      BreakContinueStack.back().BreakCount += PGO.getCurrentRegionCount();
      PGO.setCurrentRegionUnreachable();
      RecordNextStmtCount = true;
    }

    void VisitContinueStmt(const ContinueStmt *S) {
      RecordStmtCount(S);
      assert(!BreakContinueStack.empty() && "continue stmt not in a loop!");
      BreakContinueStack.back().ContinueCount += PGO.getCurrentRegionCount();
      PGO.setCurrentRegionUnreachable();
      RecordNextStmtCount = true;
    }

    void VisitWhileStmt(const WhileStmt *S) {
      RecordStmtCount(S);
      RegionCounter Cnt(PGO, S);
      BreakContinueStack.push_back(BreakContinue());
      // Visit the body region first so the break/continue adjustments can be
      // included when visiting the condition.
      Cnt.beginRegion();
      (*CountMap)[S->getBody()] = PGO.getCurrentRegionCount();
      Visit(S->getBody());
      Cnt.adjustForControlFlow();

      // ...then go back and propagate counts through the condition. The count
      // at the start of the condition is the sum of the incoming edges,
      // the backedge from the end of the loop body, and the edges from
      // continue statements.
      BreakContinue BC = BreakContinueStack.pop_back_val();
      Cnt.setCurrentRegionCount(Cnt.getParentCount() +
                                Cnt.getAdjustedCount() + BC.ContinueCount);
      (*CountMap)[S->getCond()] = PGO.getCurrentRegionCount();
      Visit(S->getCond());
      Cnt.adjustForControlFlow();
      Cnt.applyAdjustmentsToRegion(BC.BreakCount + BC.ContinueCount);
      RecordNextStmtCount = true;
    }

    void VisitDoStmt(const DoStmt *S) {
      RecordStmtCount(S);
      RegionCounter Cnt(PGO, S);
      BreakContinueStack.push_back(BreakContinue());
      Cnt.beginRegion(/*AddIncomingFallThrough=*/true);
      (*CountMap)[S->getBody()] = PGO.getCurrentRegionCount();
      Visit(S->getBody());
      Cnt.adjustForControlFlow();

      BreakContinue BC = BreakContinueStack.pop_back_val();
      // The count at the start of the condition is equal to the count at the
      // end of the body. The adjusted count does not include either the
      // fall-through count coming into the loop or the continue count, so add
      // both of those separately. This is coincidentally the same equation as
      // with while loops but for different reasons.
      Cnt.setCurrentRegionCount(Cnt.getParentCount() +
                                Cnt.getAdjustedCount() + BC.ContinueCount);
      (*CountMap)[S->getCond()] = PGO.getCurrentRegionCount();
      Visit(S->getCond());
      Cnt.adjustForControlFlow();
      Cnt.applyAdjustmentsToRegion(BC.BreakCount + BC.ContinueCount);
      RecordNextStmtCount = true;
    }

    void VisitForStmt(const ForStmt *S) {
      RecordStmtCount(S);
      if (S->getInit())
        Visit(S->getInit());
      RegionCounter Cnt(PGO, S);
      BreakContinueStack.push_back(BreakContinue());
      // Visit the body region first. (This is basically the same as a while
      // loop; see further comments in VisitWhileStmt.)
      Cnt.beginRegion();
      (*CountMap)[S->getBody()] = PGO.getCurrentRegionCount();
      Visit(S->getBody());
      Cnt.adjustForControlFlow();

      // The increment is essentially part of the body but it needs to include
      // the count for all the continue statements.
      if (S->getInc()) {
        Cnt.setCurrentRegionCount(PGO.getCurrentRegionCount() +
                                  BreakContinueStack.back().ContinueCount);
        (*CountMap)[S->getInc()] = PGO.getCurrentRegionCount();
        Visit(S->getInc());
        Cnt.adjustForControlFlow();
      }

      BreakContinue BC = BreakContinueStack.pop_back_val();

      // ...then go back and propagate counts through the condition.
      if (S->getCond()) {
        Cnt.setCurrentRegionCount(Cnt.getParentCount() +
                                  Cnt.getAdjustedCount() +
                                  BC.ContinueCount);
        (*CountMap)[S->getCond()] = PGO.getCurrentRegionCount();
        Visit(S->getCond());
        Cnt.adjustForControlFlow();
      }
      Cnt.applyAdjustmentsToRegion(BC.BreakCount + BC.ContinueCount);
      RecordNextStmtCount = true;
    }

    void VisitCXXForRangeStmt(const CXXForRangeStmt *S) {
      RecordStmtCount(S);
      Visit(S->getRangeStmt());
      Visit(S->getBeginEndStmt());
      RegionCounter Cnt(PGO, S);
      BreakContinueStack.push_back(BreakContinue());
      // Visit the body region first. (This is basically the same as a while
      // loop; see further comments in VisitWhileStmt.)
      Cnt.beginRegion();
      (*CountMap)[S->getLoopVarStmt()] = PGO.getCurrentRegionCount();
      Visit(S->getLoopVarStmt());
      Visit(S->getBody());
      Cnt.adjustForControlFlow();

      // The increment is essentially part of the body but it needs to include
      // the count for all the continue statements.
      Cnt.setCurrentRegionCount(PGO.getCurrentRegionCount() +
                                BreakContinueStack.back().ContinueCount);
      (*CountMap)[S->getInc()] = PGO.getCurrentRegionCount();
      Visit(S->getInc());
      Cnt.adjustForControlFlow();

      BreakContinue BC = BreakContinueStack.pop_back_val();

      // ...then go back and propagate counts through the condition.
      Cnt.setCurrentRegionCount(Cnt.getParentCount() +
                                Cnt.getAdjustedCount() +
                                BC.ContinueCount);
      (*CountMap)[S->getCond()] = PGO.getCurrentRegionCount();
      Visit(S->getCond());
      Cnt.adjustForControlFlow();
      Cnt.applyAdjustmentsToRegion(BC.BreakCount + BC.ContinueCount);
      RecordNextStmtCount = true;
    }

    void VisitObjCForCollectionStmt(const ObjCForCollectionStmt *S) {
      RecordStmtCount(S);
      Visit(S->getElement());
      RegionCounter Cnt(PGO, S);
      BreakContinueStack.push_back(BreakContinue());
      Cnt.beginRegion();
      (*CountMap)[S->getBody()] = PGO.getCurrentRegionCount();
      Visit(S->getBody());
      BreakContinue BC = BreakContinueStack.pop_back_val();
      Cnt.adjustForControlFlow();
      Cnt.applyAdjustmentsToRegion(BC.BreakCount + BC.ContinueCount);
      RecordNextStmtCount = true;
    }

    void VisitSwitchStmt(const SwitchStmt *S) {
      RecordStmtCount(S);
      Visit(S->getCond());
      PGO.setCurrentRegionUnreachable();
      BreakContinueStack.push_back(BreakContinue());
      Visit(S->getBody());
      // If the switch is inside a loop, add the continue counts.
      BreakContinue BC = BreakContinueStack.pop_back_val();
      if (!BreakContinueStack.empty())
        BreakContinueStack.back().ContinueCount += BC.ContinueCount;
      RegionCounter ExitCnt(PGO, S);
      ExitCnt.beginRegion();
      RecordNextStmtCount = true;
    }

    void VisitCaseStmt(const CaseStmt *S) {
      RecordNextStmtCount = false;
      RegionCounter Cnt(PGO, S);
      Cnt.beginRegion(/*AddIncomingFallThrough=*/true);
      (*CountMap)[S] = Cnt.getCount();
      RecordNextStmtCount = true;
      Visit(S->getSubStmt());
    }

    void VisitDefaultStmt(const DefaultStmt *S) {
      RecordNextStmtCount = false;
      RegionCounter Cnt(PGO, S);
      Cnt.beginRegion(/*AddIncomingFallThrough=*/true);
      (*CountMap)[S] = Cnt.getCount();
      RecordNextStmtCount = true;
      Visit(S->getSubStmt());
    }

    void VisitIfStmt(const IfStmt *S) {
      RecordStmtCount(S);
      RegionCounter Cnt(PGO, S);
      Visit(S->getCond());

      Cnt.beginRegion();
      (*CountMap)[S->getThen()] = PGO.getCurrentRegionCount();
      Visit(S->getThen());
      Cnt.adjustForControlFlow();

      if (S->getElse()) {
        Cnt.beginElseRegion();
        (*CountMap)[S->getElse()] = PGO.getCurrentRegionCount();
        Visit(S->getElse());
        Cnt.adjustForControlFlow();
      }
      Cnt.applyAdjustmentsToRegion(0);
      RecordNextStmtCount = true;
    }

    void VisitCXXTryStmt(const CXXTryStmt *S) {
      RecordStmtCount(S);
      Visit(S->getTryBlock());
      for (unsigned I = 0, E = S->getNumHandlers(); I < E; ++I)
        Visit(S->getHandler(I));
      RegionCounter Cnt(PGO, S);
      Cnt.beginRegion();
      RecordNextStmtCount = true;
    }

    void VisitCXXCatchStmt(const CXXCatchStmt *S) {
      RecordNextStmtCount = false;
      RegionCounter Cnt(PGO, S);
      Cnt.beginRegion();
      (*CountMap)[S] = PGO.getCurrentRegionCount();
      Visit(S->getHandlerBlock());
    }

    void VisitConditionalOperator(const ConditionalOperator *E) {
      RecordStmtCount(E);
      RegionCounter Cnt(PGO, E);
      Visit(E->getCond());

      Cnt.beginRegion();
      (*CountMap)[E->getTrueExpr()] = PGO.getCurrentRegionCount();
      Visit(E->getTrueExpr());
      Cnt.adjustForControlFlow();

      Cnt.beginElseRegion();
      (*CountMap)[E->getFalseExpr()] = PGO.getCurrentRegionCount();
      Visit(E->getFalseExpr());
      Cnt.adjustForControlFlow();

      Cnt.applyAdjustmentsToRegion(0);
      RecordNextStmtCount = true;
    }

    void VisitBinLAnd(const BinaryOperator *E) {
      RecordStmtCount(E);
      RegionCounter Cnt(PGO, E);
      Visit(E->getLHS());
      Cnt.beginRegion();
      (*CountMap)[E->getRHS()] = PGO.getCurrentRegionCount();
      Visit(E->getRHS());
      Cnt.adjustForControlFlow();
      Cnt.applyAdjustmentsToRegion(0);
      RecordNextStmtCount = true;
    }

    void VisitBinLOr(const BinaryOperator *E) {
      RecordStmtCount(E);
      RegionCounter Cnt(PGO, E);
      Visit(E->getLHS());
      Cnt.beginRegion();
      (*CountMap)[E->getRHS()] = PGO.getCurrentRegionCount();
      Visit(E->getRHS());
      Cnt.adjustForControlFlow();
      Cnt.applyAdjustmentsToRegion(0);
      RecordNextStmtCount = true;
    }
  };
}

void CodeGenPGO::assignRegionCounters(const Decl *D, llvm::Function *Fn) {
  bool InstrumentRegions = CGM.getCodeGenOpts().ProfileInstrGenerate;
  llvm::ProfileDataReader *PGOReader = CGM.getPGOReader();
  if (!InstrumentRegions && !PGOReader)
    return;
  if (!D)
    return;
  setFuncName(Fn);
  mapRegionCounters(D);
  if (InstrumentRegions)
    emitCounterVariables();
  if (PGOReader) {
    loadRegionCounts(PGOReader);
    computeRegionCounts(D);
    applyFunctionAttributes(PGOReader, Fn);
  }
}

void CodeGenPGO::mapRegionCounters(const Decl *D) {
  RegionCounterMap = new llvm::DenseMap<const Stmt*, unsigned>();
  MapRegionCounters Walker(RegionCounterMap);
  if (const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(D))
    Walker.VisitFunctionDecl(FD);
  else if (const ObjCMethodDecl *MD = dyn_cast_or_null<ObjCMethodDecl>(D))
    Walker.VisitObjCMethodDecl(MD);
  else if (const BlockDecl *BD = dyn_cast_or_null<BlockDecl>(D))
    Walker.VisitBlockDecl(BD);
  NumRegionCounters = Walker.NextCounter;
}

void CodeGenPGO::computeRegionCounts(const Decl *D) {
  StmtCountMap = new llvm::DenseMap<const Stmt*, uint64_t>();
  ComputeRegionCounts Walker(StmtCountMap, *this);
  if (const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(D))
    Walker.VisitFunctionDecl(FD);
  else if (const ObjCMethodDecl *MD = dyn_cast_or_null<ObjCMethodDecl>(D))
    Walker.VisitObjCMethodDecl(MD);
  else if (const BlockDecl *BD = dyn_cast_or_null<BlockDecl>(D))
    Walker.VisitBlockDecl(BD);
}

void CodeGenPGO::applyFunctionAttributes(llvm::ProfileDataReader *PGOReader,
                                         llvm::Function *Fn) {
  if (!haveRegionCounts())
    return;

  uint64_t MaxFunctionCount = PGOReader->getMaximumFunctionCount();
  uint64_t FunctionCount = getRegionCount(0);
  if (FunctionCount >= (uint64_t)(0.3 * (double)MaxFunctionCount))
    // Turn on InlineHint attribute for hot functions.
    // FIXME: 30% is from preliminary tuning on SPEC, it may not be optimal.
    Fn->addFnAttr(llvm::Attribute::InlineHint);
  else if (FunctionCount <= (uint64_t)(0.01 * (double)MaxFunctionCount))
    // Turn on Cold attribute for cold functions.
    // FIXME: 1% is from preliminary tuning on SPEC, it may not be optimal.
    Fn->addFnAttr(llvm::Attribute::Cold);
}

void CodeGenPGO::emitCounterVariables() {
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  llvm::ArrayType *CounterTy = llvm::ArrayType::get(llvm::Type::getInt64Ty(Ctx),
                                                    NumRegionCounters);
  RegionCounters =
    new llvm::GlobalVariable(CGM.getModule(), CounterTy, false,
                             llvm::GlobalVariable::PrivateLinkage,
                             llvm::Constant::getNullValue(CounterTy),
                             "__llvm_pgo_ctr");
}

void CodeGenPGO::emitCounterIncrement(CGBuilderTy &Builder, unsigned Counter) {
  if (!RegionCounters)
    return;
  llvm::Value *Addr =
    Builder.CreateConstInBoundsGEP2_64(RegionCounters, 0, Counter);
  llvm::Value *Count = Builder.CreateLoad(Addr, "pgocount");
  Count = Builder.CreateAdd(Count, Builder.getInt64(1));
  Builder.CreateStore(Count, Addr);
}

void CodeGenPGO::loadRegionCounts(llvm::ProfileDataReader *PGOReader) {
  // For now, ignore the counts from the PGO data file only if the number of
  // counters does not match. This could be tightened down in the future to
  // ignore counts when the input changes in various ways, e.g., by comparing a
  // hash value based on some characteristics of the input.
  RegionCounts = new std::vector<uint64_t>();
  uint64_t Hash;
  // TODO: Check for hash mismatch
  if (PGOReader->getFunctionCounts(getFuncName(), Hash, *RegionCounts) ||
      RegionCounts->size() != NumRegionCounters) {
    delete RegionCounts;
    RegionCounts = 0;
  }
}

void CodeGenPGO::destroyRegionCounters() {
  if (RegionCounterMap != 0)
    delete RegionCounterMap;
  if (StmtCountMap != 0)
    delete StmtCountMap;
  if (RegionCounts != 0)
    delete RegionCounts;
}

/// \brief Calculate what to divide by to scale weights.
///
/// Given the maximum weight, calculate a divisor that will scale all the
/// weights to strictly less than UINT32_MAX.
static uint64_t calculateWeightScale(uint64_t MaxWeight) {
  return MaxWeight < UINT32_MAX ? 1 : MaxWeight / UINT32_MAX + 1;
}

/// \brief Scale an individual branch weight (and add 1).
///
/// Scale a 64-bit weight down to 32-bits using \c Scale.
///
/// According to Laplace's Rule of Succession, it is better to compute the
/// weight based on the count plus 1, so universally add 1 to the value.
///
/// \pre \c Scale was calculated by \a calculateWeightScale() with a weight no
/// greater than \c Weight.
static uint32_t scaleBranchWeight(uint64_t Weight, uint64_t Scale) {
  assert(Scale && "scale by 0?");
  uint64_t Scaled = Weight / Scale + 1;
  assert(Scaled <= UINT32_MAX && "overflow 32-bits");
  return Scaled;
}

llvm::MDNode *CodeGenPGO::createBranchWeights(uint64_t TrueCount,
                                              uint64_t FalseCount) {
  // Check for empty weights.
  if (!TrueCount && !FalseCount)
    return 0;

  // Calculate how to scale down to 32-bits.
  uint64_t Scale = calculateWeightScale(std::max(TrueCount, FalseCount));

  llvm::MDBuilder MDHelper(CGM.getLLVMContext());
  return MDHelper.createBranchWeights(scaleBranchWeight(TrueCount, Scale),
                                      scaleBranchWeight(FalseCount, Scale));
}

llvm::MDNode *CodeGenPGO::createBranchWeights(ArrayRef<uint64_t> Weights) {
  // We need at least two elements to create meaningful weights.
  if (Weights.size() < 2)
    return 0;

  // Calculate how to scale down to 32-bits.
  uint64_t Scale = calculateWeightScale(*std::max_element(Weights.begin(),
                                                          Weights.end()));

  SmallVector<uint32_t, 16> ScaledWeights;
  ScaledWeights.reserve(Weights.size());
  for (uint64_t W : Weights)
    ScaledWeights.push_back(scaleBranchWeight(W, Scale));

  llvm::MDBuilder MDHelper(CGM.getLLVMContext());
  return MDHelper.createBranchWeights(ScaledWeights);
}

llvm::MDNode *CodeGenPGO::createLoopWeights(const Stmt *Cond,
                                            RegionCounter &Cnt) {
  if (!haveRegionCounts())
    return 0;
  uint64_t LoopCount = Cnt.getCount();
  uint64_t CondCount = 0;
  bool Found = getStmtCount(Cond, CondCount);
  assert(Found && "missing expected loop condition count");
  (void)Found;
  if (CondCount == 0)
    return 0;
  return createBranchWeights(LoopCount,
                             std::max(CondCount, LoopCount) - LoopCount);
}
