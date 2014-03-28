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
#include "llvm/Support/FileSystem.h"

using namespace clang;
using namespace CodeGen;

static void ReportBadPGOData(CodeGenModule &CGM, const char *Message) {
  DiagnosticsEngine &Diags = CGM.getDiags();
  unsigned diagID = Diags.getCustomDiagID(DiagnosticsEngine::Error, "%0");
  Diags.Report(diagID) << Message;
}

PGOProfileData::PGOProfileData(CodeGenModule &CGM, std::string Path)
  : CGM(CGM) {
  if (llvm::MemoryBuffer::getFile(Path, DataBuffer)) {
    ReportBadPGOData(CGM, "failed to open pgo data file");
    return;
  }

  if (DataBuffer->getBufferSize() > std::numeric_limits<unsigned>::max()) {
    ReportBadPGOData(CGM, "pgo data file too big");
    return;
  }

  // Scan through the data file and map each function to the corresponding
  // file offset where its counts are stored.
  const char *BufferStart = DataBuffer->getBufferStart();
  const char *BufferEnd = DataBuffer->getBufferEnd();
  const char *CurPtr = BufferStart;
  uint64_t MaxCount = 0;
  while (CurPtr < BufferEnd) {
    // Read the function name.
    const char *FuncStart = CurPtr;
    // For Objective-C methods, the name may include whitespace, so search
    // backward from the end of the line to find the space that separates the
    // name from the number of counters. (This is a temporary hack since we are
    // going to completely replace this file format in the near future.)
    CurPtr = strchr(CurPtr, '\n');
    if (!CurPtr) {
      ReportBadPGOData(CGM, "pgo data file has malformed function entry");
      return;
    }
    StringRef FuncName(FuncStart, CurPtr - FuncStart);

    // Skip over the function hash.
    CurPtr = strchr(++CurPtr, '\n');
    if (!CurPtr) {
      ReportBadPGOData(CGM, "pgo data file is missing the function hash");
      return;
    }

    // Read the number of counters.
    char *EndPtr;
    unsigned NumCounters = strtol(++CurPtr, &EndPtr, 10);
    if (EndPtr == CurPtr || *EndPtr != '\n' || NumCounters <= 0) {
      ReportBadPGOData(CGM, "pgo data file has unexpected number of counters");
      return;
    }
    CurPtr = EndPtr;

    // Read function count.
    uint64_t Count = strtoll(CurPtr, &EndPtr, 10);
    if (EndPtr == CurPtr || *EndPtr != '\n') {
      ReportBadPGOData(CGM, "pgo-data file has bad count value");
      return;
    }
    CurPtr = EndPtr; // Point to '\n'.
    FunctionCounts[FuncName] = Count;
    MaxCount = Count > MaxCount ? Count : MaxCount;

    // There is one line for each counter; skip over those lines.
    // Since function count is already read, we start the loop from 1.
    for (unsigned N = 1; N < NumCounters; ++N) {
      CurPtr = strchr(++CurPtr, '\n');
      if (!CurPtr) {
        ReportBadPGOData(CGM, "pgo data file is missing some counter info");
        return;
      }
    }

    // Skip over the blank line separating functions.
    CurPtr += 2;

    DataOffsets[FuncName] = FuncStart - BufferStart;
  }
  MaxFunctionCount = MaxCount;
}

bool PGOProfileData::getFunctionCounts(StringRef FuncName, uint64_t &FuncHash,
                                       std::vector<uint64_t> &Counts) {
  // Find the relevant section of the pgo-data file.
  llvm::StringMap<unsigned>::const_iterator OffsetIter =
    DataOffsets.find(FuncName);
  if (OffsetIter == DataOffsets.end())
    return true;
  const char *CurPtr = DataBuffer->getBufferStart() + OffsetIter->getValue();

  // Skip over the function name.
  CurPtr = strchr(CurPtr, '\n');
  assert(CurPtr && "pgo-data has corrupted function entry");

  char *EndPtr;
  // Read the function hash.
  FuncHash = strtoll(++CurPtr, &EndPtr, 10);
  assert(EndPtr != CurPtr && *EndPtr == '\n' &&
         "pgo-data file has corrupted function hash");
  CurPtr = EndPtr;

  // Read the number of counters.
  unsigned NumCounters = strtol(++CurPtr, &EndPtr, 10);
  assert(EndPtr != CurPtr && *EndPtr == '\n' && NumCounters > 0 &&
         "pgo-data file has corrupted number of counters");
  CurPtr = EndPtr;

  Counts.reserve(NumCounters);

  for (unsigned N = 0; N < NumCounters; ++N) {
    // Read the count value.
    uint64_t Count = strtoll(CurPtr, &EndPtr, 10);
    if (EndPtr == CurPtr || *EndPtr != '\n') {
      ReportBadPGOData(CGM, "pgo-data file has bad count value");
      return true;
    }
    Counts.push_back(Count);
    CurPtr = EndPtr + 1;
  }

  // Make sure the number of counters matches up.
  if (Counts.size() != NumCounters) {
    ReportBadPGOData(CGM, "pgo-data file has inconsistent counters");
    return true;
  }

  return false;
}

void CodeGenPGO::setFuncName(llvm::Function *Fn) {
  RawFuncName = Fn->getName();

  // Function names may be prefixed with a binary '1' to indicate
  // that the backend should not modify the symbols due to any platform
  // naming convention. Do not include that '1' in the PGO profile name.
  if (RawFuncName[0] == '\1')
    RawFuncName = RawFuncName.substr(1);

  if (!Fn->hasLocalLinkage()) {
    PrefixedFuncName.reset(new std::string(RawFuncName));
    return;
  }

  // For local symbols, prepend the main file name to distinguish them.
  // Do not include the full path in the file name since there's no guarantee
  // that it will stay the same, e.g., if the files are checked out from
  // version control in different locations.
  PrefixedFuncName.reset(new std::string(CGM.getCodeGenOpts().MainFileName));
  if (PrefixedFuncName->empty())
    PrefixedFuncName->assign("<unknown>");
  PrefixedFuncName->append(":");
  PrefixedFuncName->append(RawFuncName);
}

static llvm::Function *getRegisterFunc(CodeGenModule &CGM) {
  return CGM.getModule().getFunction("__llvm_profile_register_functions");
}

static llvm::BasicBlock *getOrInsertRegisterBB(CodeGenModule &CGM) {
  // Don't do this for Darwin.  compiler-rt uses linker magic.
  if (CGM.getTarget().getTriple().isOSDarwin())
    return nullptr;

  // Only need to insert this once per module.
  if (llvm::Function *RegisterF = getRegisterFunc(CGM))
    return &RegisterF->getEntryBlock();

  // Construct the function.
  auto *VoidTy = llvm::Type::getVoidTy(CGM.getLLVMContext());
  auto *RegisterFTy = llvm::FunctionType::get(VoidTy, false);
  auto *RegisterF = llvm::Function::Create(RegisterFTy,
                                           llvm::GlobalValue::InternalLinkage,
                                           "__llvm_profile_register_functions",
                                           &CGM.getModule());
  RegisterF->setUnnamedAddr(true);
  if (CGM.getCodeGenOpts().DisableRedZone)
    RegisterF->addFnAttr(llvm::Attribute::NoRedZone);

  // Construct and return the entry block.
  auto *BB = llvm::BasicBlock::Create(CGM.getLLVMContext(), "", RegisterF);
  CGBuilderTy Builder(BB);
  Builder.CreateRetVoid();
  return BB;
}

static llvm::Constant *getOrInsertRuntimeRegister(CodeGenModule &CGM) {
  auto *VoidTy = llvm::Type::getVoidTy(CGM.getLLVMContext());
  auto *VoidPtrTy = llvm::Type::getInt8PtrTy(CGM.getLLVMContext());
  auto *RuntimeRegisterTy = llvm::FunctionType::get(VoidTy, VoidPtrTy, false);
  return CGM.getModule().getOrInsertFunction("__llvm_profile_register_function",
                                             RuntimeRegisterTy);
}

static bool isMachO(const CodeGenModule &CGM) {
  return CGM.getTarget().getTriple().isOSBinFormatMachO();
}

static StringRef getCountersSection(const CodeGenModule &CGM) {
  return isMachO(CGM) ? "__DATA,__llvm_prf_cnts" : "__llvm_prf_cnts";
}

static StringRef getNameSection(const CodeGenModule &CGM) {
  return isMachO(CGM) ? "__DATA,__llvm_prf_names" : "__llvm_prf_names";
}

static StringRef getDataSection(const CodeGenModule &CGM) {
  return isMachO(CGM) ? "__DATA,__llvm_prf_data" : "__llvm_prf_data";
}

llvm::GlobalVariable *CodeGenPGO::buildDataVar() {
  // Create name variable.
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  auto *VarName = llvm::ConstantDataArray::getString(Ctx, getFuncName(),
                                                     false);
  auto *Name = new llvm::GlobalVariable(CGM.getModule(), VarName->getType(),
                                        true, VarLinkage, VarName,
                                        getFuncVarName("name"));
  Name->setSection(getNameSection(CGM));
  Name->setAlignment(1);

  // Create data variable.
  auto *Int32Ty = llvm::Type::getInt32Ty(Ctx);
  auto *Int64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *Int8PtrTy = llvm::Type::getInt8PtrTy(Ctx);
  auto *Int64PtrTy = llvm::Type::getInt64PtrTy(Ctx);
  llvm::Type *DataTypes[] = {
    Int32Ty, Int32Ty, Int64Ty, Int8PtrTy, Int64PtrTy
  };
  auto *DataTy = llvm::StructType::get(Ctx, makeArrayRef(DataTypes));
  llvm::Constant *DataVals[] = {
    llvm::ConstantInt::get(Int32Ty, getFuncName().size()),
    llvm::ConstantInt::get(Int32Ty, NumRegionCounters),
    llvm::ConstantInt::get(Int64Ty, FunctionHash),
    llvm::ConstantExpr::getBitCast(Name, Int8PtrTy),
    llvm::ConstantExpr::getBitCast(RegionCounters, Int64PtrTy)
  };
  auto *Data =
    new llvm::GlobalVariable(CGM.getModule(), DataTy, true, VarLinkage,
                             llvm::ConstantStruct::get(DataTy, DataVals),
                             getFuncVarName("data"));

  // All the data should be packed into an array in its own section.
  Data->setSection(getDataSection(CGM));
  Data->setAlignment(8);

  // Make sure the data doesn't get deleted.
  CGM.addUsedGlobal(Data);
  return Data;
}

void CodeGenPGO::emitInstrumentationData() {
  if (!CGM.getCodeGenOpts().ProfileInstrGenerate)
    return;

  // Build the data.
  auto *Data = buildDataVar();

  // Register the data.
  auto *RegisterBB = getOrInsertRegisterBB(CGM);
  if (!RegisterBB)
    return;
  CGBuilderTy Builder(RegisterBB->getTerminator());
  auto *VoidPtrTy = llvm::Type::getInt8PtrTy(CGM.getLLVMContext());
  Builder.CreateCall(getOrInsertRuntimeRegister(CGM),
                     Builder.CreateBitCast(Data, VoidPtrTy));
}

llvm::Function *CodeGenPGO::emitInitialization(CodeGenModule &CGM) {
  if (!CGM.getCodeGenOpts().ProfileInstrGenerate)
    return nullptr;

  // Only need to create this once per module.
  if (CGM.getModule().getFunction("__llvm_profile_init"))
    return nullptr;

  // Get the function to call at initialization.
  llvm::Constant *RegisterF = getRegisterFunc(CGM);
  if (!RegisterF)
    return nullptr;

  // Create the initialization function.
  auto *VoidTy = llvm::Type::getVoidTy(CGM.getLLVMContext());
  auto *F = llvm::Function::Create(llvm::FunctionType::get(VoidTy, false),
                                   llvm::GlobalValue::InternalLinkage,
                                   "__llvm_profile_init", &CGM.getModule());
  F->setUnnamedAddr(true);
  F->addFnAttr(llvm::Attribute::NoInline);
  if (CGM.getCodeGenOpts().DisableRedZone)
    F->addFnAttr(llvm::Attribute::NoRedZone);

  // Add the basic block and the necessary calls.
  CGBuilderTy Builder(llvm::BasicBlock::Create(CGM.getLLVMContext(), "", F));
  Builder.CreateCall(RegisterF);
  Builder.CreateRetVoid();

  return F;
}

namespace {
  /// A StmtVisitor that fills a map of statements to PGO counters.
  struct MapRegionCounters : public ConstStmtVisitor<MapRegionCounters> {
    /// The next counter value to assign.
    unsigned NextCounter;
    /// The map of statements to counters.
    llvm::DenseMap<const Stmt *, unsigned> &CounterMap;

    MapRegionCounters(llvm::DenseMap<const Stmt *, unsigned> &CounterMap)
        : NextCounter(0), CounterMap(CounterMap) {}

    void VisitChildren(const Stmt *S) {
      for (Stmt::const_child_range I = S->children(); I; ++I)
        if (*I)
         this->Visit(*I);
    }
    void VisitStmt(const Stmt *S) { VisitChildren(S); }

    /// Assign a counter to track entry to the function body.
    void VisitFunctionDecl(const FunctionDecl *S) {
      CounterMap[S->getBody()] = NextCounter++;
      Visit(S->getBody());
    }
    void VisitObjCMethodDecl(const ObjCMethodDecl *S) {
      CounterMap[S->getBody()] = NextCounter++;
      Visit(S->getBody());
    }
    void VisitBlockDecl(const BlockDecl *S) {
      CounterMap[S->getBody()] = NextCounter++;
      Visit(S->getBody());
    }
    /// Assign a counter to track the block following a label.
    void VisitLabelStmt(const LabelStmt *S) {
      CounterMap[S] = NextCounter++;
      Visit(S->getSubStmt());
    }
    /// Assign a counter for the body of a while loop.
    void VisitWhileStmt(const WhileStmt *S) {
      CounterMap[S] = NextCounter++;
      Visit(S->getCond());
      Visit(S->getBody());
    }
    /// Assign a counter for the body of a do-while loop.
    void VisitDoStmt(const DoStmt *S) {
      CounterMap[S] = NextCounter++;
      Visit(S->getBody());
      Visit(S->getCond());
    }
    /// Assign a counter for the body of a for loop.
    void VisitForStmt(const ForStmt *S) {
      CounterMap[S] = NextCounter++;
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
      CounterMap[S] = NextCounter++;
      Visit(S->getRangeStmt());
      Visit(S->getBeginEndStmt());
      Visit(S->getCond());
      Visit(S->getLoopVarStmt());
      Visit(S->getBody());
      Visit(S->getInc());
    }
    /// Assign a counter for the body of a for-collection loop.
    void VisitObjCForCollectionStmt(const ObjCForCollectionStmt *S) {
      CounterMap[S] = NextCounter++;
      Visit(S->getElement());
      Visit(S->getBody());
    }
    /// Assign a counter for the exit block of the switch statement.
    void VisitSwitchStmt(const SwitchStmt *S) {
      CounterMap[S] = NextCounter++;
      Visit(S->getCond());
      Visit(S->getBody());
    }
    /// Assign a counter for a particular case in a switch. This counts jumps
    /// from the switch header as well as fallthrough from the case before this
    /// one.
    void VisitCaseStmt(const CaseStmt *S) {
      CounterMap[S] = NextCounter++;
      Visit(S->getSubStmt());
    }
    /// Assign a counter for the default case of a switch statement. The count
    /// is the number of branches from the loop header to the default, and does
    /// not include fallthrough from previous cases. If we have multiple
    /// conditional branch blocks from the switch instruction to the default
    /// block, as with large GNU case ranges, this is the counter for the last
    /// edge in that series, rather than the first.
    void VisitDefaultStmt(const DefaultStmt *S) {
      CounterMap[S] = NextCounter++;
      Visit(S->getSubStmt());
    }
    /// Assign a counter for the "then" part of an if statement. The count for
    /// the "else" part, if it exists, will be calculated from this counter.
    void VisitIfStmt(const IfStmt *S) {
      CounterMap[S] = NextCounter++;
      Visit(S->getCond());
      Visit(S->getThen());
      if (S->getElse())
        Visit(S->getElse());
    }
    /// Assign a counter for the continuation block of a C++ try statement.
    void VisitCXXTryStmt(const CXXTryStmt *S) {
      CounterMap[S] = NextCounter++;
      Visit(S->getTryBlock());
      for (unsigned I = 0, E = S->getNumHandlers(); I < E; ++I)
        Visit(S->getHandler(I));
    }
    /// Assign a counter for a catch statement's handler block.
    void VisitCXXCatchStmt(const CXXCatchStmt *S) {
      CounterMap[S] = NextCounter++;
      Visit(S->getHandlerBlock());
    }
    /// Assign a counter for the "true" part of a conditional operator. The
    /// count in the "false" part will be calculated from this counter.
    void VisitConditionalOperator(const ConditionalOperator *E) {
      CounterMap[E] = NextCounter++;
      Visit(E->getCond());
      Visit(E->getTrueExpr());
      Visit(E->getFalseExpr());
    }
    /// Assign a counter for the right hand side of a logical and operator.
    void VisitBinLAnd(const BinaryOperator *E) {
      CounterMap[E] = NextCounter++;
      Visit(E->getLHS());
      Visit(E->getRHS());
    }
    /// Assign a counter for the right hand side of a logical or operator.
    void VisitBinLOr(const BinaryOperator *E) {
      CounterMap[E] = NextCounter++;
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
    llvm::DenseMap<const Stmt *, uint64_t> &CountMap;

    /// BreakContinueStack - Keep counts of breaks and continues inside loops. 
    struct BreakContinue {
      uint64_t BreakCount;
      uint64_t ContinueCount;
      BreakContinue() : BreakCount(0), ContinueCount(0) {}
    };
    SmallVector<BreakContinue, 8> BreakContinueStack;

    ComputeRegionCounts(llvm::DenseMap<const Stmt *, uint64_t> &CountMap,
                        CodeGenPGO &PGO)
        : PGO(PGO), RecordNextStmtCount(false), CountMap(CountMap) {}

    void RecordStmtCount(const Stmt *S) {
      if (RecordNextStmtCount) {
        CountMap[S] = PGO.getCurrentRegionCount();
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
      CountMap[S->getBody()] = PGO.getCurrentRegionCount();
      Visit(S->getBody());
    }

    void VisitObjCMethodDecl(const ObjCMethodDecl *S) {
      RegionCounter Cnt(PGO, S->getBody());
      Cnt.beginRegion();
      CountMap[S->getBody()] = PGO.getCurrentRegionCount();
      Visit(S->getBody());
    }

    void VisitBlockDecl(const BlockDecl *S) {
      RegionCounter Cnt(PGO, S->getBody());
      Cnt.beginRegion();
      CountMap[S->getBody()] = PGO.getCurrentRegionCount();
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
      CountMap[S] = PGO.getCurrentRegionCount();
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
      CountMap[S->getBody()] = PGO.getCurrentRegionCount();
      Visit(S->getBody());
      Cnt.adjustForControlFlow();

      // ...then go back and propagate counts through the condition. The count
      // at the start of the condition is the sum of the incoming edges,
      // the backedge from the end of the loop body, and the edges from
      // continue statements.
      BreakContinue BC = BreakContinueStack.pop_back_val();
      Cnt.setCurrentRegionCount(Cnt.getParentCount() +
                                Cnt.getAdjustedCount() + BC.ContinueCount);
      CountMap[S->getCond()] = PGO.getCurrentRegionCount();
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
      CountMap[S->getBody()] = PGO.getCurrentRegionCount();
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
      CountMap[S->getCond()] = PGO.getCurrentRegionCount();
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
      CountMap[S->getBody()] = PGO.getCurrentRegionCount();
      Visit(S->getBody());
      Cnt.adjustForControlFlow();

      // The increment is essentially part of the body but it needs to include
      // the count for all the continue statements.
      if (S->getInc()) {
        Cnt.setCurrentRegionCount(PGO.getCurrentRegionCount() +
                                  BreakContinueStack.back().ContinueCount);
        CountMap[S->getInc()] = PGO.getCurrentRegionCount();
        Visit(S->getInc());
        Cnt.adjustForControlFlow();
      }

      BreakContinue BC = BreakContinueStack.pop_back_val();

      // ...then go back and propagate counts through the condition.
      if (S->getCond()) {
        Cnt.setCurrentRegionCount(Cnt.getParentCount() +
                                  Cnt.getAdjustedCount() +
                                  BC.ContinueCount);
        CountMap[S->getCond()] = PGO.getCurrentRegionCount();
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
      CountMap[S->getLoopVarStmt()] = PGO.getCurrentRegionCount();
      Visit(S->getLoopVarStmt());
      Visit(S->getBody());
      Cnt.adjustForControlFlow();

      // The increment is essentially part of the body but it needs to include
      // the count for all the continue statements.
      Cnt.setCurrentRegionCount(PGO.getCurrentRegionCount() +
                                BreakContinueStack.back().ContinueCount);
      CountMap[S->getInc()] = PGO.getCurrentRegionCount();
      Visit(S->getInc());
      Cnt.adjustForControlFlow();

      BreakContinue BC = BreakContinueStack.pop_back_val();

      // ...then go back and propagate counts through the condition.
      Cnt.setCurrentRegionCount(Cnt.getParentCount() +
                                Cnt.getAdjustedCount() +
                                BC.ContinueCount);
      CountMap[S->getCond()] = PGO.getCurrentRegionCount();
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
      CountMap[S->getBody()] = PGO.getCurrentRegionCount();
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
      CountMap[S] = Cnt.getCount();
      RecordNextStmtCount = true;
      Visit(S->getSubStmt());
    }

    void VisitDefaultStmt(const DefaultStmt *S) {
      RecordNextStmtCount = false;
      RegionCounter Cnt(PGO, S);
      Cnt.beginRegion(/*AddIncomingFallThrough=*/true);
      CountMap[S] = Cnt.getCount();
      RecordNextStmtCount = true;
      Visit(S->getSubStmt());
    }

    void VisitIfStmt(const IfStmt *S) {
      RecordStmtCount(S);
      RegionCounter Cnt(PGO, S);
      Visit(S->getCond());

      Cnt.beginRegion();
      CountMap[S->getThen()] = PGO.getCurrentRegionCount();
      Visit(S->getThen());
      Cnt.adjustForControlFlow();

      if (S->getElse()) {
        Cnt.beginElseRegion();
        CountMap[S->getElse()] = PGO.getCurrentRegionCount();
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
      CountMap[S] = PGO.getCurrentRegionCount();
      Visit(S->getHandlerBlock());
    }

    void VisitConditionalOperator(const ConditionalOperator *E) {
      RecordStmtCount(E);
      RegionCounter Cnt(PGO, E);
      Visit(E->getCond());

      Cnt.beginRegion();
      CountMap[E->getTrueExpr()] = PGO.getCurrentRegionCount();
      Visit(E->getTrueExpr());
      Cnt.adjustForControlFlow();

      Cnt.beginElseRegion();
      CountMap[E->getFalseExpr()] = PGO.getCurrentRegionCount();
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
      CountMap[E->getRHS()] = PGO.getCurrentRegionCount();
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
      CountMap[E->getRHS()] = PGO.getCurrentRegionCount();
      Visit(E->getRHS());
      Cnt.adjustForControlFlow();
      Cnt.applyAdjustmentsToRegion(0);
      RecordNextStmtCount = true;
    }
  };
}

static void emitRuntimeHook(CodeGenModule &CGM) {
  LLVM_CONSTEXPR const char *RuntimeVarName = "__llvm_profile_runtime";
  LLVM_CONSTEXPR const char *RuntimeUserName = "__llvm_profile_runtime_user";
  if (CGM.getModule().getGlobalVariable(RuntimeVarName))
    return;

  // Declare the runtime hook.
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  auto *Int32Ty = llvm::Type::getInt32Ty(Ctx);
  auto *Var = new llvm::GlobalVariable(CGM.getModule(), Int32Ty, false,
                                       llvm::GlobalValue::ExternalLinkage,
                                       nullptr, RuntimeVarName);

  // Make a function that uses it.
  auto *User = llvm::Function::Create(llvm::FunctionType::get(Int32Ty, false),
                                      llvm::GlobalValue::LinkOnceODRLinkage,
                                      RuntimeUserName, &CGM.getModule());
  User->addFnAttr(llvm::Attribute::NoInline);
  if (CGM.getCodeGenOpts().DisableRedZone)
    User->addFnAttr(llvm::Attribute::NoRedZone);
  CGBuilderTy Builder(llvm::BasicBlock::Create(CGM.getLLVMContext(), "", User));
  auto *Load = Builder.CreateLoad(Var);
  Builder.CreateRet(Load);

  // Create a use of the function.  Now the definition of the runtime variable
  // should get pulled in, along with any static initializears.
  CGM.addUsedGlobal(User);
}

void CodeGenPGO::assignRegionCounters(const Decl *D, llvm::Function *Fn) {
  bool InstrumentRegions = CGM.getCodeGenOpts().ProfileInstrGenerate;
  PGOProfileData *PGOData = CGM.getPGOData();
  if (!InstrumentRegions && !PGOData)
    return;
  if (!D)
    return;
  setFuncName(Fn);

  // Set the linkage for variables based on the function linkage.  Usually, we
  // want to match it, but available_externally and extern_weak both have the
  // wrong semantics.
  VarLinkage = Fn->getLinkage();
  switch (VarLinkage) {
  case llvm::GlobalValue::ExternalWeakLinkage:
    VarLinkage = llvm::GlobalValue::LinkOnceAnyLinkage;
    break;
  case llvm::GlobalValue::AvailableExternallyLinkage:
    VarLinkage = llvm::GlobalValue::LinkOnceODRLinkage;
    break;
  default:
    break;
  }

  mapRegionCounters(D);
  if (InstrumentRegions) {
    emitRuntimeHook(CGM);
    emitCounterVariables();
  }
  if (PGOData) {
    loadRegionCounts(PGOData);
    computeRegionCounts(D);
    applyFunctionAttributes(PGOData, Fn);
  }
}

void CodeGenPGO::mapRegionCounters(const Decl *D) {
  RegionCounterMap.reset(new llvm::DenseMap<const Stmt *, unsigned>);
  MapRegionCounters Walker(*RegionCounterMap);
  if (const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(D))
    Walker.VisitFunctionDecl(FD);
  else if (const ObjCMethodDecl *MD = dyn_cast_or_null<ObjCMethodDecl>(D))
    Walker.VisitObjCMethodDecl(MD);
  else if (const BlockDecl *BD = dyn_cast_or_null<BlockDecl>(D))
    Walker.VisitBlockDecl(BD);
  NumRegionCounters = Walker.NextCounter;
  // FIXME: The number of counters isn't sufficient for the hash
  FunctionHash = NumRegionCounters;
}

void CodeGenPGO::computeRegionCounts(const Decl *D) {
  StmtCountMap.reset(new llvm::DenseMap<const Stmt *, uint64_t>);
  ComputeRegionCounts Walker(*StmtCountMap, *this);
  if (const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(D))
    Walker.VisitFunctionDecl(FD);
  else if (const ObjCMethodDecl *MD = dyn_cast_or_null<ObjCMethodDecl>(D))
    Walker.VisitObjCMethodDecl(MD);
  else if (const BlockDecl *BD = dyn_cast_or_null<BlockDecl>(D))
    Walker.VisitBlockDecl(BD);
}

void CodeGenPGO::applyFunctionAttributes(PGOProfileData *PGOData,
                                         llvm::Function *Fn) {
  if (!haveRegionCounts())
    return;

  uint64_t MaxFunctionCount = PGOData->getMaximumFunctionCount();
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
    new llvm::GlobalVariable(CGM.getModule(), CounterTy, false, VarLinkage,
                             llvm::Constant::getNullValue(CounterTy),
                             getFuncVarName("counters"));
  RegionCounters->setAlignment(8);
  RegionCounters->setSection(getCountersSection(CGM));
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

void CodeGenPGO::loadRegionCounts(PGOProfileData *PGOData) {
  // For now, ignore the counts from the PGO data file only if the number of
  // counters does not match. This could be tightened down in the future to
  // ignore counts when the input changes in various ways, e.g., by comparing a
  // hash value based on some characteristics of the input.
  RegionCounts.reset(new std::vector<uint64_t>);
  uint64_t Hash;
  if (PGOData->getFunctionCounts(getFuncName(), Hash, *RegionCounts) ||
      Hash != FunctionHash || RegionCounts->size() != NumRegionCounters)
    RegionCounts.reset();
}

void CodeGenPGO::destroyRegionCounters() {
  RegionCounterMap.reset();
  StmtCountMap.reset();
  RegionCounts.reset();
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
    return nullptr;

  // Calculate how to scale down to 32-bits.
  uint64_t Scale = calculateWeightScale(std::max(TrueCount, FalseCount));

  llvm::MDBuilder MDHelper(CGM.getLLVMContext());
  return MDHelper.createBranchWeights(scaleBranchWeight(TrueCount, Scale),
                                      scaleBranchWeight(FalseCount, Scale));
}

llvm::MDNode *CodeGenPGO::createBranchWeights(ArrayRef<uint64_t> Weights) {
  // We need at least two elements to create meaningful weights.
  if (Weights.size() < 2)
    return nullptr;

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
    return nullptr;
  uint64_t LoopCount = Cnt.getCount();
  uint64_t CondCount = 0;
  bool Found = getStmtCount(Cond, CondCount);
  assert(Found && "missing expected loop condition count");
  (void)Found;
  if (CondCount == 0)
    return nullptr;
  return createBranchWeights(LoopCount,
                             std::max(CondCount, LoopCount) - LoopCount);
}
