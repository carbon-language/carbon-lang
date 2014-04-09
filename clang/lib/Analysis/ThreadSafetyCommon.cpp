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

#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "clang/Analysis/Analyses/ThreadSafetyCommon.h"
#include "clang/Analysis/Analyses/ThreadSafetyTIL.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <vector>


namespace clang {
namespace threadSafety {

typedef SExprBuilder::CallingContext CallingContext;


til::SExpr *SExprBuilder::lookupStmt(const Stmt *S) {
  if (!SMap)
    return nullptr;
  auto It = SMap->find(S);
  if (It != SMap->end())
    return It->second;
  return nullptr;
}

void SExprBuilder::insertStmt(const Stmt *S, til::Variable *V) {
  SMap->insert(std::make_pair(S, V));
}


// Translate a clang statement or expression to a TIL expression.
// Also performs substitution of variables; Ctx provides the context.
// Dispatches on the type of S.
til::SExpr *SExprBuilder::translate(const Stmt *S, CallingContext *Ctx) {
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

  case BO_Assign:
    return new (Arena)
        til::Store(translate(BO->getLHS(), Ctx), translate(BO->getRHS(), Ctx));

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
  til::SExpr *E0 = translate(CE->getSubExpr(), Ctx);

  clang::CastKind K = CE->getCastKind();
  switch (K) {
  case CK_LValueToRValue:
    return new (Arena) til::Load(E0);

  case CK_NoOp:
  case CK_DerivedToBase:
  case CK_UncheckedDerivedToBase:
  case CK_ArrayToPointerDecay:
  case CK_FunctionToPointerDecay:
    return E0;

  default:
    return new (Arena) til::Cast(K, E0);
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

// Build a complete SCFG from a clang CFG.
class SCFGBuilder {
  void addStatement(til::SExpr* E, const Stmt *S) {
    if (!E)
      return;
    if (til::ThreadSafetyTIL::isTrivial(E))
      return;

    til::Variable *V = new (Arena) til::Variable(til::Variable::VK_Let, E);
    V->setID(CurrentBlockID, CurrentVarID++);
    CurrentBB->addInstr(V);
    if (S)
      BuildEx.insertStmt(S, V);
  }

public:
  // Enter the CFG for Decl D, and perform any initial setup operations.
  void enterCFG(CFG *Cfg, const NamedDecl *D, const CFGBlock *First) {
    Scfg = new (Arena) til::SCFG(Arena, Cfg->getNumBlockIDs());
    CallCtx = new SExprBuilder::CallingContext(D);
  }

  // Enter a CFGBlock.
  void enterCFGBlock(const CFGBlock *B) {
    CurrentBB = new (Arena) til::BasicBlock(Arena, 0, B->size());
    CurrentBB->setBlockID(CurrentBlockID);
    CurrentVarID = 0;
    Scfg->add(CurrentBB);
  }

  // Process an ordinary statement.
  void handleStatement(const Stmt *S) {
    til::SExpr *E = BuildEx.translate(S, CallCtx);
    addStatement(E, S);
  }

  // Process a destructor call
  void handleDestructorCall(const VarDecl *VD, const CXXDestructorDecl *DD) {
    til::SExpr *Sf = new (Arena) til::LiteralPtr(VD);
    til::SExpr *Dr = new (Arena) til::LiteralPtr(DD);
    til::SExpr *Ap = new (Arena) til::Apply(Dr, Sf);
    til::SExpr *E = new (Arena) til::Call(Ap);
    addStatement(E, nullptr);
  }

  // Process a successor edge.
  void handleSuccessor(const CFGBlock *Succ) {}

  // Process a successor back edge to a previously visited block.
  void handleSuccessorBackEdge(const CFGBlock *Succ) {}

  // Leave a CFGBlock.
  void exitCFGBlock(const CFGBlock *B) {
    CurrentBlockID++;
    CurrentBB = 0;
  }

  // Leave the CFG, and perform any final cleanup operations.
  void exitCFG(const CFGBlock *Last) {
    delete CallCtx;
    CallCtx = nullptr;
  }

  SCFGBuilder(til::MemRegionRef A)
      : Arena(A), Scfg(nullptr), CurrentBB(nullptr), CurrentBlockID(0),
        CurrentVarID(0), CallCtx(nullptr),
        SMap(new SExprBuilder::StatementMap()), BuildEx(A, SMap) {}
  ~SCFGBuilder() { delete SMap; }

  til::SCFG *getCFG() const { return Scfg; }

private:
  til::MemRegionRef Arena;
  til::SCFG *Scfg;
  til::BasicBlock *CurrentBB;
  unsigned CurrentBlockID;
  unsigned CurrentVarID;

  SExprBuilder::CallingContext *CallCtx;
  SExprBuilder::StatementMap *SMap;
  SExprBuilder BuildEx;
};



class LLVMPrinter :
    public til::PrettyPrinter<LLVMPrinter, llvm::raw_ostream> {
};


void printSCFG(CFGWalker &walker) {
  llvm::BumpPtrAllocator Bpa;
  til::MemRegionRef Arena(&Bpa);
  SCFGBuilder builder(Arena);
  // CFGVisitor visitor;
  walker.walk(builder);
  LLVMPrinter::print(builder.getCFG(), llvm::errs());
}



} // end namespace threadSafety

} // end namespace clang
