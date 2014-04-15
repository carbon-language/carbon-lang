//===- ThreadSafetyCommon.cpp ----------------------------------*- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the interfaces declared in ThreadSafetyCommon.h
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/ThreadSafetyCommon.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "clang/Analysis/Analyses/ThreadSafetyTIL.h"
#include "clang/Analysis/Analyses/ThreadSafetyTraverse.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <climits>
#include <vector>


namespace clang {
namespace threadSafety {

typedef SExprBuilder::CallingContext CallingContext;


til::SExpr *SExprBuilder::lookupStmt(const Stmt *S) {
  auto It = SMap.find(S);
  if (It != SMap.end())
    return It->second;
  return nullptr;
}

void SExprBuilder::insertStmt(const Stmt *S, til::Variable *V) {
  SMap.insert(std::make_pair(S, V));
}


til::SCFG *SExprBuilder::buildCFG(CFGWalker &Walker) {
  Walker.walk(*this);
  return Scfg;
}


// Translate a clang statement or expression to a TIL expression.
// Also performs substitution of variables; Ctx provides the context.
// Dispatches on the type of S.
til::SExpr *SExprBuilder::translate(const Stmt *S, CallingContext *Ctx) {
  if (!S)
    return nullptr;

  // Check if S has already been translated and cached.
  // This handles the lookup of SSA names for DeclRefExprs here.
  if (til::SExpr *E = lookupStmt(S))
    return E;

  switch (S->getStmtClass()) {
  case Stmt::DeclRefExprClass:
    return translateDeclRefExpr(cast<DeclRefExpr>(S), Ctx);
  case Stmt::CXXThisExprClass:
    return translateCXXThisExpr(cast<CXXThisExpr>(S), Ctx);
  case Stmt::MemberExprClass:
    return translateMemberExpr(cast<MemberExpr>(S), Ctx);
  case Stmt::CallExprClass:
    return translateCallExpr(cast<CallExpr>(S), Ctx);
  case Stmt::CXXMemberCallExprClass:
    return translateCXXMemberCallExpr(cast<CXXMemberCallExpr>(S), Ctx);
  case Stmt::CXXOperatorCallExprClass:
    return translateCXXOperatorCallExpr(cast<CXXOperatorCallExpr>(S), Ctx);
  case Stmt::UnaryOperatorClass:
    return translateUnaryOperator(cast<UnaryOperator>(S), Ctx);
  case Stmt::BinaryOperatorClass:
    return translateBinaryOperator(cast<BinaryOperator>(S), Ctx);

  case Stmt::ArraySubscriptExprClass:
    return translateArraySubscriptExpr(cast<ArraySubscriptExpr>(S), Ctx);
  case Stmt::ConditionalOperatorClass:
    return translateConditionalOperator(cast<ConditionalOperator>(S), Ctx);
  case Stmt::BinaryConditionalOperatorClass:
    return translateBinaryConditionalOperator(
             cast<BinaryConditionalOperator>(S), Ctx);

  // We treat these as no-ops
  case Stmt::ParenExprClass:
    return translate(cast<ParenExpr>(S)->getSubExpr(), Ctx);
  case Stmt::ExprWithCleanupsClass:
    return translate(cast<ExprWithCleanups>(S)->getSubExpr(), Ctx);
  case Stmt::CXXBindTemporaryExprClass:
    return translate(cast<CXXBindTemporaryExpr>(S)->getSubExpr(), Ctx);

  // Collect all literals
  case Stmt::CharacterLiteralClass:
  case Stmt::CXXNullPtrLiteralExprClass:
  case Stmt::GNUNullExprClass:
  case Stmt::CXXBoolLiteralExprClass:
  case Stmt::FloatingLiteralClass:
  case Stmt::ImaginaryLiteralClass:
  case Stmt::IntegerLiteralClass:
  case Stmt::StringLiteralClass:
  case Stmt::ObjCStringLiteralClass:
    return new (Arena) til::Literal(cast<Expr>(S));

  case Stmt::DeclStmtClass:
    return translateDeclStmt(cast<DeclStmt>(S), Ctx);
  default:
    break;
  }
  if (const CastExpr *CE = dyn_cast<CastExpr>(S))
    return translateCastExpr(CE, Ctx);

  return new (Arena) til::Undefined(S);
}


til::SExpr *SExprBuilder::translateDeclRefExpr(const DeclRefExpr *DRE,
                                               CallingContext *Ctx) {
  const ValueDecl *VD = cast<ValueDecl>(DRE->getDecl()->getCanonicalDecl());

  // Function parameters require substitution and/or renaming.
  if (const ParmVarDecl *PV = dyn_cast_or_null<ParmVarDecl>(VD)) {
    const FunctionDecl *FD =
        cast<FunctionDecl>(PV->getDeclContext())->getCanonicalDecl();
    unsigned I = PV->getFunctionScopeIndex();

    if (Ctx && Ctx->FunArgs && FD == Ctx->AttrDecl->getCanonicalDecl()) {
      // Substitute call arguments for references to function parameters
      assert(I < Ctx->NumArgs);
      return translate(Ctx->FunArgs[I], Ctx->Prev);
    }
    // Map the param back to the param of the original function declaration
    // for consistent comparisons.
    VD = FD->getParamDecl(I);
  }

  // For non-local variables, treat it as a referenced to a named object.
  return new (Arena) til::LiteralPtr(VD);
}


til::SExpr *SExprBuilder::translateCXXThisExpr(const CXXThisExpr *TE,
                                               CallingContext *Ctx) {
  // Substitute for 'this'
  if (Ctx && Ctx->SelfArg)
    return translate(Ctx->SelfArg, Ctx->Prev);
  assert(SelfVar && "We have no variable for 'this'!");
  return SelfVar;
}


til::SExpr *SExprBuilder::translateMemberExpr(const MemberExpr *ME,
                                              CallingContext *Ctx) {
  til::SExpr *E = translate(ME->getBase(), Ctx);
  E = new (Arena) til::SApply(E);
  return new (Arena) til::Project(E, ME->getMemberDecl());
}


til::SExpr *SExprBuilder::translateCallExpr(const CallExpr *CE,
                                            CallingContext *Ctx) {
  // TODO -- Lock returned
  til::SExpr *E = translate(CE->getCallee(), Ctx);
  for (const auto *Arg : CE->arguments()) {
    til::SExpr *A = translate(Arg, Ctx);
    E = new (Arena) til::Apply(E, A);
  }
  return new (Arena) til::Call(E, CE);
}


til::SExpr *SExprBuilder::translateCXXMemberCallExpr(
    const CXXMemberCallExpr *ME, CallingContext *Ctx) {
  return translateCallExpr(cast<CallExpr>(ME), Ctx);
}


til::SExpr *SExprBuilder::translateCXXOperatorCallExpr(
    const CXXOperatorCallExpr *OCE, CallingContext *Ctx) {
  return translateCallExpr(cast<CallExpr>(OCE), Ctx);
}


til::SExpr *SExprBuilder::translateUnaryOperator(const UnaryOperator *UO,
                                                 CallingContext *Ctx) {
  switch (UO->getOpcode()) {
  case UO_PostInc:
  case UO_PostDec:
  case UO_PreInc:
  case UO_PreDec:
    return new (Arena) til::Undefined(UO);

  // We treat these as no-ops
  case UO_AddrOf:
  case UO_Deref:
  case UO_Plus:
    return translate(UO->getSubExpr(), Ctx);

  case UO_Minus:
  case UO_Not:
  case UO_LNot:
  case UO_Real:
  case UO_Imag:
  case UO_Extension:
    return new (Arena)
        til::UnaryOp(UO->getOpcode(), translate(UO->getSubExpr(), Ctx));
  }
  return new (Arena) til::Undefined(UO);
}


til::SExpr *SExprBuilder::translateBinaryOperator(const BinaryOperator *BO,
                                                  CallingContext *Ctx) {
  switch (BO->getOpcode()) {
  case BO_PtrMemD:
  case BO_PtrMemI:
    return new (Arena) til::Undefined(BO);

  case BO_Mul:
  case BO_Div:
  case BO_Rem:
  case BO_Add:
  case BO_Sub:
  case BO_Shl:
  case BO_Shr:
  case BO_LT:
  case BO_GT:
  case BO_LE:
  case BO_GE:
  case BO_EQ:
  case BO_NE:
  case BO_And:
  case BO_Xor:
  case BO_Or:
  case BO_LAnd:
  case BO_LOr:
    return new (Arena)
        til::BinaryOp(BO->getOpcode(), translate(BO->getLHS(), Ctx),
                      translate(BO->getRHS(), Ctx));

  case BO_Assign: {
    const Expr *LHS = BO->getLHS();
    if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(LHS)) {
      const Expr *RHS = BO->getRHS();
      til::SExpr *E1 = translate(RHS, Ctx);
      return updateVarDecl(DRE->getDecl(), E1);
    }
    til::SExpr *E0 = translate(LHS, Ctx);
    til::SExpr *E1 = translate(BO->getRHS(), Ctx);
    return new (Arena) til::Store(E0, E1);
  }
  case BO_MulAssign:
  case BO_DivAssign:
  case BO_RemAssign:
  case BO_AddAssign:
  case BO_SubAssign:
  case BO_ShlAssign:
  case BO_ShrAssign:
  case BO_AndAssign:
  case BO_XorAssign:
  case BO_OrAssign:
    return new (Arena) til::Undefined(BO);

  case BO_Comma:
    // TODO: handle LHS
    return translate(BO->getRHS(), Ctx);
  }

  return new (Arena) til::Undefined(BO);
}


til::SExpr *SExprBuilder::translateCastExpr(const CastExpr *CE,
                                            CallingContext *Ctx) {
  clang::CastKind K = CE->getCastKind();
  switch (K) {
  case CK_LValueToRValue: {
    if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(CE->getSubExpr())) {
      til::SExpr *E0 = lookupVarDecl(DRE->getDecl());
      if (E0)
        return E0;
    }
    til::SExpr *E0 = translate(CE->getSubExpr(), Ctx);
    return new (Arena) til::Load(E0);
  }
  case CK_NoOp:
  case CK_DerivedToBase:
  case CK_UncheckedDerivedToBase:
  case CK_ArrayToPointerDecay:
  case CK_FunctionToPointerDecay: {
    til::SExpr *E0 = translate(CE->getSubExpr(), Ctx);
    return E0;
  }
  default: {
    til::SExpr *E0 = translate(CE->getSubExpr(), Ctx);
    return new (Arena) til::Cast(K, E0);
  }
  }
}


til::SExpr *
SExprBuilder::translateArraySubscriptExpr(const ArraySubscriptExpr *E,
                                          CallingContext *Ctx) {
  return new (Arena) til::Undefined(E);
}


til::SExpr *
SExprBuilder::translateConditionalOperator(const ConditionalOperator *C,
                                           CallingContext *Ctx) {
  return new (Arena) til::Undefined(C);
}


til::SExpr *SExprBuilder::translateBinaryConditionalOperator(
    const BinaryConditionalOperator *C, CallingContext *Ctx) {
  return new (Arena) til::Undefined(C);
}


til::SExpr *
SExprBuilder::translateDeclStmt(const DeclStmt *S, CallingContext *Ctx) {
  DeclGroupRef DGrp = S->getDeclGroup();
  for (DeclGroupRef::iterator I = DGrp.begin(), E = DGrp.end(); I != E; ++I) {
    if (VarDecl *VD = dyn_cast_or_null<VarDecl>(*I)) {
      Expr *E = VD->getInit();
      til::SExpr* SE = translate(E, Ctx);

      // Add local variables with trivial type to the variable map
      QualType T = VD->getType();
      if (T.isTrivialType(VD->getASTContext())) {
        return addVarDecl(VD, SE);
      }
      else {
        // TODO: add alloca
      }
    }
  }
  return nullptr;
}


// If (E) is non-trivial, then add it to the current basic block, and
// update the statement map so that S refers to E.  Returns a new variable
// that refers to E.
// If E is trivial returns E.
til::SExpr *SExprBuilder::addStatement(til::SExpr* E, const Stmt *S,
                                       const ValueDecl *VD) {
  if (!E)
    return nullptr;
  if (til::ThreadSafetyTIL::isTrivial(E))
    return E;

  til::Variable *V = new (Arena) til::Variable(E, VD);
  V->setID(CurrentBlockID, CurrentVarID++);
  CurrentBB->addInstr(V);
  if (S)
    insertStmt(S, V);
  return V;
}


// Returns the current value of VD, if known, and nullptr otherwise.
til::SExpr *SExprBuilder::lookupVarDecl(const ValueDecl *VD) {
  auto It = IdxMap.find(VD);
  if (It != IdxMap.end())
    return CurrentNameMap[It->second].second;
  return nullptr;
}


// if E is a til::Variable, update its clangDecl.
inline void maybeUpdateVD(til::SExpr *E, const ValueDecl *VD) {
  if (!E)
    return;
  if (til::Variable *V = dyn_cast<til::Variable>(E)) {
    if (!V->clangDecl())
      V->setClangDecl(VD);
  }
}

// Adds a new variable declaration.
til::SExpr *SExprBuilder::addVarDecl(const ValueDecl *VD, til::SExpr *E) {
  maybeUpdateVD(E, VD);
  IdxMap.insert(std::make_pair(VD, CurrentNameMap.size()));
  CurrentNameMap.makeWritable();
  CurrentNameMap.push_back(std::make_pair(VD, E));
  return E;
}


// Updates a current variable declaration.  (E.g. by assignment)
til::SExpr *SExprBuilder::updateVarDecl(const ValueDecl *VD, til::SExpr *E) {
  maybeUpdateVD(E, VD);
  auto It = IdxMap.find(VD);
  if (It == IdxMap.end()) {
    til::SExpr *Ptr = new (Arena) til::LiteralPtr(VD);
    til::SExpr *St  = new (Arena) til::Store(Ptr, E);
    return St;
  }
  CurrentNameMap.makeWritable();
  CurrentNameMap.elem(It->second).second = E;
  return E;
}


// Merge values from Map into the current entry map.
void SExprBuilder::mergeEntryMap(NameVarMap Map) {
  assert(CurrentBlockInfo && "Not processing a block!");

  if (!CurrentNameMap.valid()) {
    // Steal Map, using copy-on-write.
    CurrentNameMap = std::move(Map);
    return;
  }
  if (CurrentNameMap.sameAs(Map))
    return;  // Easy merge: maps from different predecessors are unchanged.

  unsigned ESz = CurrentNameMap.size();
  unsigned MSz = Map.size();
  unsigned Sz = std::max(ESz, MSz);
  bool W = CurrentNameMap.writable();
  for (unsigned i=0; i<Sz; ++i) {
    if (CurrentNameMap[i].first != Map[i].first) {
      if (!W)
        CurrentNameMap.makeWritable();
      CurrentNameMap.downsize(i);
      break;
    }
    if (CurrentNameMap[i].second != Map[i].second) {
      til::Variable *V =
        dyn_cast<til::Variable>(CurrentNameMap[i].second);
      if (V && V->getBlockID() == CurrentBB->blockID()) {
        // We already have a Phi node, so add the new variable.
        til::Phi *Ph = dyn_cast<til::Phi>(V->definition());
        assert(Ph && "Expecting Phi node.");
        Ph->values()[CurrentArgIndex] = Map[i].second;
      }
      else {
        if (!W)
          CurrentNameMap.makeWritable();
        unsigned NPreds = CurrentBB->numPredecessors();
        assert(CurrentArgIndex > 0 && CurrentArgIndex < NPreds);

        // Make a new phi node.  All phi args up to the current index must
        // be the same, and equal to the current NameMap value.
        auto *Ph = new (Arena) til::Phi(Arena, NPreds);
        Ph->values().setValues(NPreds, nullptr);
        for (unsigned PIdx = 0; PIdx < CurrentArgIndex; ++PIdx)
          Ph->values()[PIdx] = CurrentNameMap[i].second;
        Ph->values()[CurrentArgIndex] = Map[i].second;

        // Add phi node to current basic block.
        auto *Var = new (Arena) til::Variable(Ph, CurrentNameMap[i].first);
        Var->setID(CurrentBlockID, CurrentVarID++);
        CurrentBB->addArgument(Var);
        CurrentNameMap.elem(i).second = Var;
      }
    }
  }
  if (ESz > MSz) {
    if (!W)
      CurrentNameMap.makeWritable();
    CurrentNameMap.downsize(Map.size());
  }
}



void SExprBuilder::enterCFG(CFG *Cfg, const NamedDecl *D,
                            const CFGBlock *First) {
  // Perform initial setup operations.
  unsigned NBlocks = Cfg->getNumBlockIDs();
  Scfg = new (Arena) til::SCFG(Arena, NBlocks);

  // allocate all basic blocks immediately, to handle forward references.
  BlockMap.reserve(NBlocks);
  BBInfo.resize(NBlocks);
  for (auto *B : *Cfg) {
    auto *BB = new (Arena) til::BasicBlock(Arena, 0, B->size());
    BlockMap.push_back(BB);
  }
  CallCtx = new SExprBuilder::CallingContext(D);
}



void SExprBuilder::enterCFGBlock(const CFGBlock *B) {
  // Intialize TIL basic block and add it to the CFG.
  CurrentBB = BlockMap[B->getBlockID()];
  CurrentBB->setBlockID(CurrentBlockID);
  CurrentBB->setNumPredecessors(B->pred_size());
  Scfg->add(CurrentBB);

  CurrentBlockInfo = &BBInfo[B->getBlockID()];
  CurrentVarID = 0;
  CurrentArgIndex = 0;

  assert(!CurrentNameMap.valid() && "CurrentNameMap already initialized.");
}


void SExprBuilder::handlePredecessor(const CFGBlock *Pred) {
  // Compute CurrentNameMap on entry from ExitMaps of predecessors

  BlockInfo *PredInfo = &BBInfo[Pred->getBlockID()];
  assert(PredInfo->SuccessorsToProcess > 0);

  if (--PredInfo->SuccessorsToProcess == 0)
    mergeEntryMap(std::move(PredInfo->ExitMap));
  else
    mergeEntryMap(PredInfo->ExitMap.clone());

  ++CurrentArgIndex;
}


void SExprBuilder::handlePredecessorBackEdge(const CFGBlock *Pred) {
  CurrentBlockInfo->HasBackEdges = true;
}


void SExprBuilder::enterCFGBlockBody(const CFGBlock *B) { }


void SExprBuilder::handleStatement(const Stmt *S) {
  til::SExpr *E = translate(S, CallCtx);
  addStatement(E, S);
}


void SExprBuilder::handleDestructorCall(const VarDecl *VD,
                                        const CXXDestructorDecl *DD) {
  til::SExpr *Sf = new (Arena) til::LiteralPtr(VD);
  til::SExpr *Dr = new (Arena) til::LiteralPtr(DD);
  til::SExpr *Ap = new (Arena) til::Apply(Dr, Sf);
  til::SExpr *E = new (Arena) til::Call(Ap);
  addStatement(E, nullptr);
}



void SExprBuilder::exitCFGBlockBody(const CFGBlock *B) {
  unsigned N = B->succ_size();
  auto It = B->succ_begin();
  if (N == 1) {
    til::BasicBlock *BB = *It ? BlockMap[(*It)->getBlockID()] : nullptr;
    // TODO: set index
    til::SExpr *Tm = new (Arena) til::Goto(BB, 0);
    CurrentBB->setTerminator(Tm);
  }
  else if (N == 2) {
    til::SExpr *C = translate(B->getTerminatorCondition(true), CallCtx);
    til::BasicBlock *BB1 = *It ? BlockMap[(*It)->getBlockID()] : nullptr;
    ++It;
    til::BasicBlock *BB2 = *It ? BlockMap[(*It)->getBlockID()] : nullptr;
    // TODO: set conditional, set index
    til::SExpr *Tm = new (Arena) til::Branch(C, BB1, BB2);
    CurrentBB->setTerminator(Tm);
  }
}


void SExprBuilder::handleSuccessor(const CFGBlock *Succ) {
  ++CurrentBlockInfo->SuccessorsToProcess;
}


void SExprBuilder::handleSuccessorBackEdge(const CFGBlock *Succ) {

}


void SExprBuilder::exitCFGBlock(const CFGBlock *B) {
  CurrentBlockInfo->ExitMap = std::move(CurrentNameMap);
  CurrentBlockID++;
  CurrentBB = nullptr;
  CurrentBlockInfo = nullptr;
}


void SExprBuilder::exitCFG(const CFGBlock *Last) {
  CurrentBlockID = 0;
  CurrentVarID = 0;
  CurrentArgIndex = 0;
}



class LLVMPrinter : public til::PrettyPrinter<LLVMPrinter, llvm::raw_ostream> {
};


void printSCFG(CFGWalker &Walker) {
  llvm::BumpPtrAllocator Bpa;
  til::MemRegionRef Arena(&Bpa);
  SExprBuilder builder(Arena);
  til::SCFG *Cfg = builder.buildCFG(Walker);
  LLVMPrinter::print(Cfg, llvm::errs());
}



} // end namespace threadSafety

} // end namespace clang
