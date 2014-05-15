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
#include "clang/AST/DeclObjC.h"
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

// From ThreadSafetyUtil.h
std::string getSourceLiteralString(const clang::Expr *CE) {
  switch (CE->getStmtClass()) {
    case Stmt::IntegerLiteralClass:
      return cast<IntegerLiteral>(CE)->getValue().toString(10, true);
    case Stmt::StringLiteralClass: {
      std::string ret("\"");
      ret += cast<StringLiteral>(CE)->getString();
      ret += "\"";
      return ret;
    }
    case Stmt::CharacterLiteralClass:
    case Stmt::CXXNullPtrLiteralExprClass:
    case Stmt::GNUNullExprClass:
    case Stmt::CXXBoolLiteralExprClass:
    case Stmt::FloatingLiteralClass:
    case Stmt::ImaginaryLiteralClass:
    case Stmt::ObjCStringLiteralClass:
    default:
      return "#lit";
  }
}

namespace til {

// Return true if E is a variable that points to an incomplete Phi node.
static bool isIncompleteVar(const SExpr *E) {
  if (const auto *V = dyn_cast<Variable>(E)) {
    if (const auto *Ph = dyn_cast<Phi>(V->definition()))
      return Ph->status() == Phi::PH_Incomplete;
  }
  return false;
}

}  // end namespace til


typedef SExprBuilder::CallingContext CallingContext;


til::SExpr *SExprBuilder::lookupStmt(const Stmt *S) {
  auto It = SMap.find(S);
  if (It != SMap.end())
    return It->second;
  return nullptr;
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
  case Stmt::CompoundAssignOperatorClass:
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
    return new (Arena)
      til::UnaryOp(til::UOP_Minus, translate(UO->getSubExpr(), Ctx));
  case UO_Not:
    return new (Arena)
      til::UnaryOp(til::UOP_BitNot, translate(UO->getSubExpr(), Ctx));
  case UO_LNot:
    return new (Arena)
      til::UnaryOp(til::UOP_LogicNot, translate(UO->getSubExpr(), Ctx));

  // Currently unsupported
  case UO_Real:
  case UO_Imag:
  case UO_Extension:
    return new (Arena) til::Undefined(UO);
  }
  return new (Arena) til::Undefined(UO);
}


til::SExpr *SExprBuilder::translateBinOp(til::TIL_BinaryOpcode Op,
                                         const BinaryOperator *BO,
                                         CallingContext *Ctx, bool Reverse) {
   til::SExpr *E0 = translate(BO->getLHS(), Ctx);
   til::SExpr *E1 = translate(BO->getRHS(), Ctx);
   if (Reverse)
     return new (Arena) til::BinaryOp(Op, E1, E0);
   else
     return new (Arena) til::BinaryOp(Op, E0, E1);
}


til::SExpr *SExprBuilder::translateBinAssign(til::TIL_BinaryOpcode Op,
                                             const BinaryOperator *BO,
                                             CallingContext *Ctx,
                                             bool Assign) {
  const Expr *LHS = BO->getLHS();
  const Expr *RHS = BO->getRHS();
  til::SExpr *E0 = translate(LHS, Ctx);
  til::SExpr *E1 = translate(RHS, Ctx);

  const ValueDecl *VD = nullptr;
  til::SExpr *CV = nullptr;
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(LHS)) {
    VD = DRE->getDecl();
    CV = lookupVarDecl(VD);
  }

  if (!Assign) {
    til::SExpr *Arg = CV ? CV : new (Arena) til::Load(E0);
    E1 = new (Arena) til::BinaryOp(Op, Arg, E1);
    E1 = addStatement(E1, nullptr, VD);
  }
  if (VD && CV)
    return updateVarDecl(VD, E1);
  return new (Arena) til::Store(E0, E1);
}


til::SExpr *SExprBuilder::translateBinaryOperator(const BinaryOperator *BO,
                                                  CallingContext *Ctx) {
  switch (BO->getOpcode()) {
  case BO_PtrMemD:
  case BO_PtrMemI:
    return new (Arena) til::Undefined(BO);

  case BO_Mul:  return translateBinOp(til::BOP_Mul, BO, Ctx);
  case BO_Div:  return translateBinOp(til::BOP_Div, BO, Ctx);
  case BO_Rem:  return translateBinOp(til::BOP_Rem, BO, Ctx);
  case BO_Add:  return translateBinOp(til::BOP_Add, BO, Ctx);
  case BO_Sub:  return translateBinOp(til::BOP_Sub, BO, Ctx);
  case BO_Shl:  return translateBinOp(til::BOP_Shl, BO, Ctx);
  case BO_Shr:  return translateBinOp(til::BOP_Shr, BO, Ctx);
  case BO_LT:   return translateBinOp(til::BOP_Lt,  BO, Ctx);
  case BO_GT:   return translateBinOp(til::BOP_Lt,  BO, Ctx, true);
  case BO_LE:   return translateBinOp(til::BOP_Leq, BO, Ctx);
  case BO_GE:   return translateBinOp(til::BOP_Leq, BO, Ctx, true);
  case BO_EQ:   return translateBinOp(til::BOP_Eq,  BO, Ctx);
  case BO_NE:   return translateBinOp(til::BOP_Neq, BO, Ctx);
  case BO_And:  return translateBinOp(til::BOP_BitAnd,   BO, Ctx);
  case BO_Xor:  return translateBinOp(til::BOP_BitXor,   BO, Ctx);
  case BO_Or:   return translateBinOp(til::BOP_BitOr,    BO, Ctx);
  case BO_LAnd: return translateBinOp(til::BOP_LogicAnd, BO, Ctx);
  case BO_LOr:  return translateBinOp(til::BOP_LogicOr,  BO, Ctx);

  case BO_Assign:    return translateBinAssign(til::BOP_Eq,  BO, Ctx, true);
  case BO_MulAssign: return translateBinAssign(til::BOP_Mul, BO, Ctx);
  case BO_DivAssign: return translateBinAssign(til::BOP_Div, BO, Ctx);
  case BO_RemAssign: return translateBinAssign(til::BOP_Rem, BO, Ctx);
  case BO_AddAssign: return translateBinAssign(til::BOP_Add, BO, Ctx);
  case BO_SubAssign: return translateBinAssign(til::BOP_Sub, BO, Ctx);
  case BO_ShlAssign: return translateBinAssign(til::BOP_Shl, BO, Ctx);
  case BO_ShrAssign: return translateBinAssign(til::BOP_Shr, BO, Ctx);
  case BO_AndAssign: return translateBinAssign(til::BOP_BitAnd, BO, Ctx);
  case BO_XorAssign: return translateBinAssign(til::BOP_BitXor, BO, Ctx);
  case BO_OrAssign:  return translateBinAssign(til::BOP_BitOr,  BO, Ctx);

  case BO_Comma:
    // The clang CFG should have already processed both sides.
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
    // FIXME: handle different kinds of casts.
    til::SExpr *E0 = translate(CE->getSubExpr(), Ctx);
    return new (Arena) til::Cast(til::CAST_none, E0);
  }
  }
}


til::SExpr *
SExprBuilder::translateArraySubscriptExpr(const ArraySubscriptExpr *E,
                                          CallingContext *Ctx) {
  til::SExpr *E0 = translate(E->getBase(), Ctx);
  til::SExpr *E1 = translate(E->getIdx(), Ctx);
  auto *AA = new (Arena) til::ArrayAdd(E0, E1);
  return new (Arena) til::ArrayFirst(AA);
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
  CurrentInstructions.push_back(V);
  if (S)
    insertStmt(S, V);
  return V;
}


// Returns the current value of VD, if known, and nullptr otherwise.
til::SExpr *SExprBuilder::lookupVarDecl(const ValueDecl *VD) {
  auto It = LVarIdxMap.find(VD);
  if (It != LVarIdxMap.end()) {
    assert(CurrentLVarMap[It->second].first == VD);
    return CurrentLVarMap[It->second].second;
  }
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
  LVarIdxMap.insert(std::make_pair(VD, CurrentLVarMap.size()));
  CurrentLVarMap.makeWritable();
  CurrentLVarMap.push_back(std::make_pair(VD, E));
  return E;
}


// Updates a current variable declaration.  (E.g. by assignment)
til::SExpr *SExprBuilder::updateVarDecl(const ValueDecl *VD, til::SExpr *E) {
  maybeUpdateVD(E, VD);
  auto It = LVarIdxMap.find(VD);
  if (It == LVarIdxMap.end()) {
    til::SExpr *Ptr = new (Arena) til::LiteralPtr(VD);
    til::SExpr *St  = new (Arena) til::Store(Ptr, E);
    return St;
  }
  CurrentLVarMap.makeWritable();
  CurrentLVarMap.elem(It->second).second = E;
  return E;
}


// Make a Phi node in the current block for the i^th variable in CurrentVarMap.
// If E != null, sets Phi[CurrentBlockInfo->ArgIndex] = E.
// If E == null, this is a backedge and will be set later.
void SExprBuilder::makePhiNodeVar(unsigned i, unsigned NPreds, til::SExpr *E) {
  unsigned ArgIndex = CurrentBlockInfo->ProcessedPredecessors;
  assert(ArgIndex > 0 && ArgIndex < NPreds);

  til::Variable *V = dyn_cast<til::Variable>(CurrentLVarMap[i].second);
  if (V && V->getBlockID() == CurrentBB->blockID()) {
    // We already have a Phi node in the current block,
    // so just add the new variable to the Phi node.
    til::Phi *Ph = dyn_cast<til::Phi>(V->definition());
    assert(Ph && "Expecting Phi node.");
    if (E)
      Ph->values()[ArgIndex] = E;
    return;
  }

  // Make a new phi node: phi(..., E)
  // All phi args up to the current index are set to the current value.
  til::SExpr *CurrE = CurrentLVarMap[i].second;
  til::Phi *Ph = new (Arena) til::Phi(Arena, NPreds);
  Ph->values().setValues(NPreds, nullptr);
  for (unsigned PIdx = 0; PIdx < ArgIndex; ++PIdx)
    Ph->values()[PIdx] = CurrE;
  if (E)
    Ph->values()[ArgIndex] = E;
  // If E is from a back-edge, or either E or CurrE are incomplete, then
  // mark this node as incomplete; we may need to remove it later.
  if (!E || isIncompleteVar(E) || isIncompleteVar(CurrE)) {
    Ph->setStatus(til::Phi::PH_Incomplete);
  }

  // Add Phi node to current block, and update CurrentLVarMap[i]
  auto *Var = new (Arena) til::Variable(Ph, CurrentLVarMap[i].first);
  CurrentArguments.push_back(Var);
  if (Ph->status() == til::Phi::PH_Incomplete)
    IncompleteArgs.push_back(Var);

  CurrentLVarMap.makeWritable();
  CurrentLVarMap.elem(i).second = Var;
}


// Merge values from Map into the current variable map.
// This will construct Phi nodes in the current basic block as necessary.
void SExprBuilder::mergeEntryMap(LVarDefinitionMap Map) {
  assert(CurrentBlockInfo && "Not processing a block!");

  if (!CurrentLVarMap.valid()) {
    // Steal Map, using copy-on-write.
    CurrentLVarMap = std::move(Map);
    return;
  }
  if (CurrentLVarMap.sameAs(Map))
    return;  // Easy merge: maps from different predecessors are unchanged.

  unsigned NPreds = CurrentBB->numPredecessors();
  unsigned ESz = CurrentLVarMap.size();
  unsigned MSz = Map.size();
  unsigned Sz  = std::min(ESz, MSz);

  for (unsigned i=0; i<Sz; ++i) {
    if (CurrentLVarMap[i].first != Map[i].first) {
      // We've reached the end of variables in common.
      CurrentLVarMap.makeWritable();
      CurrentLVarMap.downsize(i);
      break;
    }
    if (CurrentLVarMap[i].second != Map[i].second)
      makePhiNodeVar(i, NPreds, Map[i].second);
  }
  if (ESz > MSz) {
    CurrentLVarMap.makeWritable();
    CurrentLVarMap.downsize(Map.size());
  }
}


// Merge a back edge into the current variable map.
// This will create phi nodes for all variables in the variable map.
void SExprBuilder::mergeEntryMapBackEdge() {
  // We don't have definitions for variables on the backedge, because we
  // haven't gotten that far in the CFG.  Thus, when encountering a back edge,
  // we conservatively create Phi nodes for all variables.  Unnecessary Phi
  // nodes will be marked as incomplete, and stripped out at the end.
  //
  // An Phi node is unnecessary if it only refers to itself and one other
  // variable, e.g. x = Phi(y, y, x)  can be reduced to x = y.

  assert(CurrentBlockInfo && "Not processing a block!");

  if (CurrentBlockInfo->HasBackEdges)
    return;
  CurrentBlockInfo->HasBackEdges = true;

  CurrentLVarMap.makeWritable();
  unsigned Sz = CurrentLVarMap.size();
  unsigned NPreds = CurrentBB->numPredecessors();

  for (unsigned i=0; i < Sz; ++i) {
    makePhiNodeVar(i, NPreds, nullptr);
  }
}


// Update the phi nodes that were initially created for a back edge
// once the variable definitions have been computed.
// I.e., merge the current variable map into the phi nodes for Blk.
void SExprBuilder::mergePhiNodesBackEdge(const CFGBlock *Blk) {
  til::BasicBlock *BB = lookupBlock(Blk);
  unsigned ArgIndex = BBInfo[Blk->getBlockID()].ProcessedPredecessors;
  assert(ArgIndex > 0 && ArgIndex < BB->numPredecessors());

  for (til::Variable *V : BB->arguments()) {
    til::Phi *Ph = dyn_cast_or_null<til::Phi>(V->definition());
    assert(Ph && "Expecting Phi Node.");
    assert(Ph->values()[ArgIndex] == nullptr && "Wrong index for back edge.");
    assert(V->clangDecl() && "No local variable for Phi node.");

    til::SExpr *E = lookupVarDecl(V->clangDecl());
    assert(E && "Couldn't find local variable for Phi node.");

    Ph->values()[ArgIndex] = E;
  }
}

void SExprBuilder::enterCFG(CFG *Cfg, const NamedDecl *D,
                            const CFGBlock *First) {
  // Perform initial setup operations.
  unsigned NBlocks = Cfg->getNumBlockIDs();
  Scfg = new (Arena) til::SCFG(Arena, NBlocks);

  // allocate all basic blocks immediately, to handle forward references.
  BBInfo.resize(NBlocks);
  BlockMap.resize(NBlocks, nullptr);
  // create map from clang blockID to til::BasicBlocks
  for (auto *B : *Cfg) {
    auto *BB = new (Arena) til::BasicBlock(Arena, 0, B->size());
    BlockMap[B->getBlockID()] = BB;
  }
  CallCtx.reset(new SExprBuilder::CallingContext(D));

  CurrentBB = lookupBlock(&Cfg->getEntry());
  auto Parms = isa<ObjCMethodDecl>(D) ? cast<ObjCMethodDecl>(D)->parameters()
                                      : cast<FunctionDecl>(D)->parameters();
  for (auto *Pm : Parms) {
    QualType T = Pm->getType();
    if (!T.isTrivialType(Pm->getASTContext()))
      continue;

    // Add parameters to local variable map.
    // FIXME: right now we emulate params with loads; that should be fixed.
    til::SExpr *Lp = new (Arena) til::LiteralPtr(Pm);
    til::SExpr *Ld = new (Arena) til::Load(Lp);
    til::SExpr *V  = addStatement(Ld, nullptr, Pm);
    addVarDecl(Pm, V);
  }
}


void SExprBuilder::enterCFGBlock(const CFGBlock *B) {
  // Intialize TIL basic block and add it to the CFG.
  CurrentBB = lookupBlock(B);
  CurrentBB->setNumPredecessors(B->pred_size());
  Scfg->add(CurrentBB);

  CurrentBlockInfo = &BBInfo[B->getBlockID()];

  // CurrentLVarMap is moved to ExitMap on block exit.
  // FIXME: the entry block will hold function parameters.
  // assert(!CurrentLVarMap.valid() && "CurrentLVarMap already initialized.");
}


void SExprBuilder::handlePredecessor(const CFGBlock *Pred) {
  // Compute CurrentLVarMap on entry from ExitMaps of predecessors

  BlockInfo *PredInfo = &BBInfo[Pred->getBlockID()];
  assert(PredInfo->UnprocessedSuccessors > 0);

  if (--PredInfo->UnprocessedSuccessors == 0)
    mergeEntryMap(std::move(PredInfo->ExitMap));
  else
    mergeEntryMap(PredInfo->ExitMap.clone());

  ++CurrentBlockInfo->ProcessedPredecessors;
}


void SExprBuilder::handlePredecessorBackEdge(const CFGBlock *Pred) {
  mergeEntryMapBackEdge();
}


void SExprBuilder::enterCFGBlockBody(const CFGBlock *B) {
  // The merge*() methods have created arguments.
  // Push those arguments onto the basic block.
  CurrentBB->arguments().reserve(
    static_cast<unsigned>(CurrentArguments.size()), Arena);
  for (auto *V : CurrentArguments)
    CurrentBB->addArgument(V);
}


void SExprBuilder::handleStatement(const Stmt *S) {
  til::SExpr *E = translate(S, CallCtx.get());
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
  CurrentBB->instructions().reserve(
    static_cast<unsigned>(CurrentInstructions.size()), Arena);
  for (auto *V : CurrentInstructions)
    CurrentBB->addInstruction(V);

  // Create an appropriate terminator
  unsigned N = B->succ_size();
  auto It = B->succ_begin();
  if (N == 1) {
    til::BasicBlock *BB = *It ? lookupBlock(*It) : nullptr;
    // TODO: set index
    til::SExpr *Tm = new (Arena) til::Goto(BB, 0);
    CurrentBB->setTerminator(Tm);
  }
  else if (N == 2) {
    til::SExpr *C = translate(B->getTerminatorCondition(true), CallCtx.get());
    til::BasicBlock *BB1 = *It ? lookupBlock(*It) : nullptr;
    ++It;
    til::BasicBlock *BB2 = *It ? lookupBlock(*It) : nullptr;
    // TODO: set conditional, set index
    til::SExpr *Tm = new (Arena) til::Branch(C, BB1, BB2);
    CurrentBB->setTerminator(Tm);
  }
}


void SExprBuilder::handleSuccessor(const CFGBlock *Succ) {
  ++CurrentBlockInfo->UnprocessedSuccessors;
}


void SExprBuilder::handleSuccessorBackEdge(const CFGBlock *Succ) {
  mergePhiNodesBackEdge(Succ);
  ++BBInfo[Succ->getBlockID()].ProcessedPredecessors;
}


void SExprBuilder::exitCFGBlock(const CFGBlock *B) {
  CurrentArguments.clear();
  CurrentInstructions.clear();
  CurrentBlockInfo->ExitMap = std::move(CurrentLVarMap);
  CurrentBB = nullptr;
  CurrentBlockInfo = nullptr;
}


void SExprBuilder::exitCFG(const CFGBlock *Last) {
  for (auto *V : IncompleteArgs) {
    til::Phi *Ph = dyn_cast<til::Phi>(V->definition());
    if (Ph && Ph->status() == til::Phi::PH_Incomplete)
      simplifyIncompleteArg(V, Ph);
  }

  CurrentArguments.clear();
  CurrentInstructions.clear();
  IncompleteArgs.clear();
}



class TILPrinter : public til::PrettyPrinter<TILPrinter, llvm::raw_ostream> {};


void printSCFG(CFGWalker &Walker) {
  llvm::BumpPtrAllocator Bpa;
  til::MemRegionRef Arena(&Bpa);
  SExprBuilder builder(Arena);
  til::SCFG *Cfg = builder.buildCFG(Walker);
  TILPrinter::print(Cfg, llvm::errs());
}



} // end namespace threadSafety

} // end namespace clang
