//==- UninitializedValues.cpp - Find Unintialized Values --------*- C++ --*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Uninitialized Values analysis for source-level CFGs.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/UninitializedValues.h"
#include "clang/Analysis/CFGStmtVisitor.h"
#include "clang/Analysis/LocalCheckers.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/AST/ASTContext.h"
#include "DataflowSolver.h"

#include "llvm/ADT/SmallPtrSet.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// Dataflow initialization logic.
//===----------------------------------------------------------------------===//      

namespace {

class RegisterDeclsAndExprs : public CFGStmtVisitor<RegisterDeclsAndExprs> {
  UninitializedValues::AnalysisDataTy& AD;
public:
  RegisterDeclsAndExprs(UninitializedValues::AnalysisDataTy& ad) :  AD(ad) {}
  
  void VisitBlockVarDecl(BlockVarDecl* VD) {
    if (AD.VMap.find(VD) == AD.VMap.end())
      AD.VMap[VD] = AD.NumDecls++;
  }
      
  void VisitDeclChain(ScopedDecl* D) {
    for (; D != NULL; D = D->getNextDeclarator())
      if (BlockVarDecl* VD = dyn_cast<BlockVarDecl>(D))
        VisitBlockVarDecl(VD);
  }
  
  void BlockStmt_VisitExpr(Expr* E) {
    if (AD.EMap.find(E) == AD.EMap.end())
      AD.EMap[E] = AD.NumBlockExprs++;
      
    Visit(E);
  }
  
  void VisitDeclRefExpr(DeclRefExpr* DR) { VisitDeclChain(DR->getDecl()); }
  void VisitDeclStmt(DeclStmt* S) { VisitDeclChain(S->getDecl()); }
  void VisitStmt(Stmt* S) { VisitChildren(S); }
  void operator()(Stmt* S) { BlockStmt_Visit(S); }
};
  
} // end anonymous namespace

void UninitializedValues::InitializeValues(const CFG& cfg) {
  RegisterDeclsAndExprs R(this->getAnalysisData());
  cfg.VisitBlockStmts(R);
}

//===----------------------------------------------------------------------===//
// Transfer functions.
//===----------------------------------------------------------------------===//      

namespace {

class TransferFuncs : public CFGStmtVisitor<TransferFuncs,bool> {
  UninitializedValues::ValTy V;
  UninitializedValues::AnalysisDataTy& AD;
  bool InitWithAssigns;
public:
  TransferFuncs(UninitializedValues::AnalysisDataTy& ad, 
                bool init_with_assigns=true) : 
    AD(ad), InitWithAssigns(init_with_assigns) {
    V.resetValues(AD);
  }
  
  UninitializedValues::ValTy& getVal() { return V; }
  
  bool VisitDeclRefExpr(DeclRefExpr* DR);
  bool VisitBinaryOperator(BinaryOperator* B);
  bool VisitUnaryOperator(UnaryOperator* U);
  bool VisitStmt(Stmt* S);
  bool VisitCallExpr(CallExpr* C);
  bool BlockStmt_VisitExpr(Expr* E);
  bool VisitDeclStmt(DeclStmt* D);
  
  BlockVarDecl* FindBlockVarDecl(Stmt* S);
  
  static inline bool Initialized() { return true; }
  static inline bool Uninitialized() { return false; }
};


bool TransferFuncs::VisitDeclRefExpr(DeclRefExpr* DR) {
  if (BlockVarDecl* VD = dyn_cast<BlockVarDecl>(DR->getDecl())) {
    if (AD.Observer) AD.Observer->ObserveDeclRefExpr(V,AD,DR,VD);
      
    return V.getBitRef(VD,AD);
  }
  else return Initialized();
}

BlockVarDecl* TransferFuncs::FindBlockVarDecl(Stmt *S) {
  for (;;) {
    if (ParenExpr* P = dyn_cast<ParenExpr>(S)) {
      S = P->getSubExpr();
      continue;
    }
    else if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(S))
      if (BlockVarDecl* VD = dyn_cast<BlockVarDecl>(DR->getDecl()))
        return VD;

    return NULL;
  }          
}

bool TransferFuncs::VisitBinaryOperator(BinaryOperator* B) {

  if (CFG::hasImplicitControlFlow(B))
    return V.getBitRef(B,AD);
  
  if (B->isAssignmentOp())
    // Get the Decl for the LHS (if any).
    if (BlockVarDecl* VD = FindBlockVarDecl(B->getLHS()))
      if(InitWithAssigns) {
        // Pseudo-hack to prevent cascade of warnings.  If the RHS uses
        // an uninitialized value, then we are already going to flag a warning
        // for the RHS, or for the root "source" of the unintialized values.  
        // Thus, propogating uninitialized doesn't make sense, since we are
        // just adding extra messages that don't
        // contribute to diagnosing the bug.  In InitWithAssigns mode
        // we unconditionally set the assigned variable to Initialized to
        // prevent Uninitialized propogation.
        return V.getBitRef(VD,AD) = Initialized();
      }
      else return V.getBitRef(VD,AD) = Visit(B->getRHS());
  
  return VisitStmt(B);
}

bool TransferFuncs::VisitDeclStmt(DeclStmt* S) {
  bool x = Initialized();
  
  for (ScopedDecl* D = S->getDecl(); D != NULL; D = D->getNextDeclarator())
    if (BlockVarDecl* VD = dyn_cast<BlockVarDecl>(D))
      if (Stmt* I = VD->getInit()) {
        x = V.getBitRef(cast<Expr>(I),AD);
        V.getBitRef(VD,AD) = x;
      }
      
  return x;
}

bool TransferFuncs::VisitCallExpr(CallExpr* C) {
  VisitChildren(C);
  return Initialized();
}

bool TransferFuncs::VisitUnaryOperator(UnaryOperator* U) {
  switch (U->getOpcode()) {
    case UnaryOperator::AddrOf:
      // For "&x", treat "x" as now being initialized.
      if (BlockVarDecl* VD = FindBlockVarDecl(U->getSubExpr()))
        V.getBitRef(VD,AD) = Initialized();
      else 
        return Visit(U->getSubExpr());

    default:
      return Visit(U->getSubExpr());
  }      
}

bool TransferFuncs::VisitStmt(Stmt* S) {
  bool x = Initialized();

  // We don't stop at the first subexpression that is Uninitialized because
  // evaluating some subexpressions may result in propogating "Uninitialized"
  // or "Initialized" to variables referenced in the other subexpressions.
  for (Stmt::child_iterator I=S->child_begin(), E=S->child_end(); I!=E; ++I)
    if (Visit(*I) == Uninitialized()) x = Uninitialized();
  
  return x;
}

bool TransferFuncs::BlockStmt_VisitExpr(Expr* E) {
  assert (AD.isTracked(E));
  return V.getBitRef(E,AD) = Visit(E);
}
  
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Merge operator.
//
//  In our transfer functions we take the approach that any
//  combination of unintialized values, e.g. Unitialized + ___ = Unitialized.
//
//  Merges take the opposite approach.
//
//  In the merge of dataflow values we prefer unsoundness, and
//  prefer false negatives to false positives.  At merges, if a value for a
//  tracked Decl is EVER initialized in any of the predecessors we treat it as
//  initialized at the confluence point.
//===----------------------------------------------------------------------===//      

namespace {
struct Merge {
  void operator()(UninitializedValues::ValTy& Dst,
                  UninitializedValues::ValTy& Src) {
    assert (Src.sizesEqual(Dst) && "BV sizes do not match.");
    Dst.DeclBV |= Src.DeclBV;
    Dst.ExprBV |= Src.ExprBV;
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Unitialized values checker.   Scan an AST and flag variable uses
//===----------------------------------------------------------------------===//      

UninitializedValues_ValueTypes::ObserverTy::~ObserverTy() {}

namespace {
class UninitializedValuesChecker : public UninitializedValues::ObserverTy {
  ASTContext &Ctx;
  Diagnostic &Diags;
  llvm::SmallPtrSet<BlockVarDecl*,10> AlreadyWarned;
  
public:
  UninitializedValuesChecker(ASTContext &ctx, Diagnostic &diags)
    : Ctx(ctx), Diags(diags) {}
    
  virtual void ObserveDeclRefExpr(UninitializedValues::ValTy& V,
                                  UninitializedValues::AnalysisDataTy& AD,
                                  DeclRefExpr* DR, BlockVarDecl* VD) {

    assert ( AD.isTracked(VD) && "Unknown VarDecl.");
    
    if (V.getBitRef(VD,AD) == TransferFuncs::Uninitialized())
      if (AlreadyWarned.insert(VD))
        Diags.Report(DR->getSourceRange().Begin(), diag::warn_uninit_val);
  }
};
} // end anonymous namespace

namespace clang {
void CheckUninitializedValues(CFG& cfg, ASTContext &Ctx, Diagnostic &Diags) {

  typedef DataflowSolver<UninitializedValues,TransferFuncs,Merge> Solver;
  
  // Compute the unitialized values information.
  UninitializedValues U;
  Solver S(U);
  S.runOnCFG(cfg);
  
  // Scan for DeclRefExprs that use uninitialized values.
  UninitializedValuesChecker Observer(Ctx,Diags);
  U.getAnalysisData().Observer = &Observer;
  S.runOnAllBlocks(cfg);
}
} // end namespace clang
