//==- UninitializedValues.cpp - Find Uninitialized Values -------*- C++ --*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Uninitialized Values analysis for source-level CFGs.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/UninitializedValues.h"
#include "clang/Analysis/Visitors/CFGRecStmtDeclVisitor.h"
#include "clang/Analysis/AnalysisDiagnostic.h"
#include "clang/AST/ASTContext.h"
#include "clang/Analysis/FlowSensitive/DataflowSolver.h"

#include "llvm/ADT/SmallPtrSet.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// Dataflow initialization logic.
//===----------------------------------------------------------------------===//

namespace {

class RegisterDecls
  : public CFGRecStmtDeclVisitor<RegisterDecls> {

  UninitializedValues::AnalysisDataTy& AD;
public:
  RegisterDecls(UninitializedValues::AnalysisDataTy& ad) :  AD(ad) {}

  void VisitVarDecl(VarDecl* VD) { AD.Register(VD); }
  CFG& getCFG() { return AD.getCFG(); }
};

} // end anonymous namespace

void UninitializedValues::InitializeValues(const CFG& cfg) {
  RegisterDecls R(getAnalysisData());
  cfg.VisitBlockStmts(R);
}

//===----------------------------------------------------------------------===//
// Transfer functions.
//===----------------------------------------------------------------------===//

namespace {
class TransferFuncs
  : public CFGStmtVisitor<TransferFuncs,bool> {

  UninitializedValues::ValTy V;
  UninitializedValues::AnalysisDataTy& AD;
public:
  TransferFuncs(UninitializedValues::AnalysisDataTy& ad) : AD(ad) {}

  UninitializedValues::ValTy& getVal() { return V; }
  CFG& getCFG() { return AD.getCFG(); }

  void SetTopValue(UninitializedValues::ValTy& X) {
    X.setDeclValues(AD);
    X.resetBlkExprValues(AD);
  }

  bool VisitDeclRefExpr(DeclRefExpr* DR);
  bool VisitBinaryOperator(BinaryOperator* B);
  bool VisitUnaryOperator(UnaryOperator* U);
  bool VisitStmt(Stmt* S);
  bool VisitCallExpr(CallExpr* C);
  bool VisitDeclStmt(DeclStmt* D);
  bool VisitConditionalOperator(ConditionalOperator* C);
  bool BlockStmt_VisitObjCForCollectionStmt(ObjCForCollectionStmt* S);

  bool Visit(Stmt *S);
  bool BlockStmt_VisitExpr(Expr* E);

  void VisitTerminator(CFGBlock* B) { }
};

static const bool Initialized = false;
static const bool Uninitialized = true;

bool TransferFuncs::VisitDeclRefExpr(DeclRefExpr* DR) {

  if (VarDecl* VD = dyn_cast<VarDecl>(DR->getDecl()))
    if (VD->isBlockVarDecl()) {

      if (AD.Observer)
        AD.Observer->ObserveDeclRefExpr(V, AD, DR, VD);

      // Pseudo-hack to prevent cascade of warnings.  If an accessed variable
      // is uninitialized, then we are already going to flag a warning for
      // this variable, which a "source" of uninitialized values.
      // We can otherwise do a full "taint" of uninitialized values.  The
      // client has both options by toggling AD.FullUninitTaint.

      if (AD.FullUninitTaint)
        return V(VD,AD);
    }

  return Initialized;
}

static VarDecl* FindBlockVarDecl(Expr* E) {

  // Blast through casts and parentheses to find any DeclRefExprs that
  // refer to a block VarDecl.

  if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(E->IgnoreParenCasts()))
    if (VarDecl* VD = dyn_cast<VarDecl>(DR->getDecl()))
      if (VD->isBlockVarDecl()) return VD;

  return NULL;
}

bool TransferFuncs::VisitBinaryOperator(BinaryOperator* B) {

  if (VarDecl* VD = FindBlockVarDecl(B->getLHS()))
    if (B->isAssignmentOp()) {
      if (B->getOpcode() == BinaryOperator::Assign)
        return V(VD,AD) = Visit(B->getRHS());
      else // Handle +=, -=, *=, etc.  We do want '&', not '&&'.
        return V(VD,AD) = Visit(B->getLHS()) & Visit(B->getRHS());
    }

  return VisitStmt(B);
}

bool TransferFuncs::VisitDeclStmt(DeclStmt* S) {
  for (DeclStmt::decl_iterator I=S->decl_begin(), E=S->decl_end(); I!=E; ++I) {
    VarDecl *VD = dyn_cast<VarDecl>(*I);
    if (VD && VD->isBlockVarDecl()) {
      if (Stmt* I = VD->getInit())
        V(VD,AD) = AD.FullUninitTaint ? V(cast<Expr>(I),AD) : Initialized;
      else {
        // Special case for declarations of array types.  For things like:
        //
        //  char x[10];
        //
        // we should treat "x" as being initialized, because the variable
        // "x" really refers to the memory block.  Clearly x[1] is
        // uninitialized, but expressions like "(char *) x" really do refer to
        // an initialized value.  This simple dataflow analysis does not reason
        // about the contents of arrays, although it could be potentially
        // extended to do so if the array were of constant size.
        if (VD->getType()->isArrayType())
          V(VD,AD) = Initialized;
        else
          V(VD,AD) = Uninitialized;
      }
    }
  }
  return Uninitialized; // Value is never consumed.
}

bool TransferFuncs::VisitCallExpr(CallExpr* C) {
  VisitChildren(C);
  return Initialized;
}

bool TransferFuncs::VisitUnaryOperator(UnaryOperator* U) {
  switch (U->getOpcode()) {
    case UnaryOperator::AddrOf: {
      VarDecl* VD = FindBlockVarDecl(U->getSubExpr());
      if (VD && VD->isBlockVarDecl())
        return V(VD,AD) = Initialized;
      break;
    }

    default:
      break;
  }

  return Visit(U->getSubExpr());
}

bool
TransferFuncs::BlockStmt_VisitObjCForCollectionStmt(ObjCForCollectionStmt* S) {
  // This represents a use of the 'collection'
  bool x = Visit(S->getCollection());

  if (x == Uninitialized)
    return Uninitialized;

  // This represents an initialization of the 'element' value.
  Stmt* Element = S->getElement();
  VarDecl* VD = 0;

  if (DeclStmt* DS = dyn_cast<DeclStmt>(Element))
    VD = cast<VarDecl>(DS->getSingleDecl());
  else {
    Expr* ElemExpr = cast<Expr>(Element)->IgnoreParens();

    // Initialize the value of the reference variable.
    if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(ElemExpr))
      VD = cast<VarDecl>(DR->getDecl());
    else
      return Visit(ElemExpr);
  }

  V(VD,AD) = Initialized;
  return Initialized;
}


bool TransferFuncs::VisitConditionalOperator(ConditionalOperator* C) {
  Visit(C->getCond());

  bool rhsResult = Visit(C->getRHS());
  // Handle the GNU extension for missing LHS.
  if (Expr *lhs = C->getLHS())
    return Visit(lhs) & rhsResult; // Yes: we want &, not &&.
  else
    return rhsResult;
}

bool TransferFuncs::VisitStmt(Stmt* S) {
  bool x = Initialized;

  // We don't stop at the first subexpression that is Uninitialized because
  // evaluating some subexpressions may result in propogating "Uninitialized"
  // or "Initialized" to variables referenced in the other subexpressions.
  for (Stmt::child_iterator I=S->child_begin(), E=S->child_end(); I!=E; ++I)
    if (*I && Visit(*I) == Uninitialized) x = Uninitialized;

  return x;
}

bool TransferFuncs::Visit(Stmt *S) {
  if (AD.isTracked(static_cast<Expr*>(S))) return V(static_cast<Expr*>(S),AD);
  else return static_cast<CFGStmtVisitor<TransferFuncs,bool>*>(this)->Visit(S);
}

bool TransferFuncs::BlockStmt_VisitExpr(Expr* E) {
  bool x = static_cast<CFGStmtVisitor<TransferFuncs,bool>*>(this)->Visit(E);
  if (AD.isTracked(E)) V(E,AD) = x;
  return x;
}

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Merge operator.
//
//  In our transfer functions we take the approach that any
//  combination of uninitialized values, e.g.
//      Uninitialized + ___ = Uninitialized.
//
//  Merges take the same approach, preferring soundness.  At a confluence point,
//  if any predecessor has a variable marked uninitialized, the value is
//  uninitialized at the confluence point.
//===----------------------------------------------------------------------===//

namespace {
  typedef StmtDeclBitVector_Types::Union Merge;
  typedef DataflowSolver<UninitializedValues,TransferFuncs,Merge> Solver;
}

//===----------------------------------------------------------------------===//
// Uninitialized values checker.   Scan an AST and flag variable uses
//===----------------------------------------------------------------------===//

UninitializedValues_ValueTypes::ObserverTy::~ObserverTy() {}

namespace {
class UninitializedValuesChecker
  : public UninitializedValues::ObserverTy {

  ASTContext &Ctx;
  Diagnostic &Diags;
  llvm::SmallPtrSet<VarDecl*,10> AlreadyWarned;

public:
  UninitializedValuesChecker(ASTContext &ctx, Diagnostic &diags)
    : Ctx(ctx), Diags(diags) {}

  virtual void ObserveDeclRefExpr(UninitializedValues::ValTy& V,
                                  UninitializedValues::AnalysisDataTy& AD,
                                  DeclRefExpr* DR, VarDecl* VD) {

    assert ( AD.isTracked(VD) && "Unknown VarDecl.");

    if (V(VD,AD) == Uninitialized)
      if (AlreadyWarned.insert(VD))
        Diags.Report(Ctx.getFullLoc(DR->getSourceRange().getBegin()),
                     diag::warn_uninit_val);
  }
};
} // end anonymous namespace

namespace clang {
void CheckUninitializedValues(CFG& cfg, ASTContext &Ctx, Diagnostic &Diags,
                              bool FullUninitTaint) {

  // Compute the uninitialized values information.
  UninitializedValues U(cfg);
  U.getAnalysisData().FullUninitTaint = FullUninitTaint;
  Solver S(U);
  S.runOnCFG(cfg);

  // Scan for DeclRefExprs that use uninitialized values.
  UninitializedValuesChecker Observer(Ctx,Diags);
  U.getAnalysisData().Observer = &Observer;
  S.runOnAllBlocks(cfg);
}
} // end namespace clang
