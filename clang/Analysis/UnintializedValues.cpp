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
#include "clang/Analysis/CFGVarDeclVisitor.h"
#include "clang/Analysis/CFGStmtVisitor.h"
#include "DataflowSolver.h"

using namespace clang;

//===--------------------------------------------------------------------===//
// Dataflow initialization logic.
//===--------------------------------------------------------------------===//      

namespace {

class RegisterDeclsAndExprs : public CFGVarDeclVisitor<RegisterDeclsAndExprs> {
  UninitializedValues::AnalysisDataTy& AD;
public:
  RegisterDeclsAndExprs(const CFG& cfg, UninitializedValues::AnalysisDataTy& ad)
                        : CFGVarDeclVisitor<RegisterDeclsAndExprs>(cfg), AD(ad)
  {}
  
  void VisitVarDecl(VarDecl* D) {
    if (AD.VMap.find(D) == AD.VMap.end())
      AD.VMap[D] = AD.NumDecls++;
  }
  
  void BlockStmt_VisitExpr(Expr* E) {
    if (AD.EMap.find(E) == AD.EMap.end())
      AD.EMap[E] = AD.NumBlockExprs++;
  }        
};
  
} // end anonymous namespace

void UninitializedValues::InitializeValues(const CFG& cfg) {
  RegisterDeclsAndExprs R(cfg,this->getAnalysisData());
  R.VisitAllDecls();    
  UninitializedValues::ValTy& V = getBlockDataMap()[&cfg.getEntry()];
  V.DeclBV.resize(getAnalysisData().NumDecls);
  V.ExprBV.resize(getAnalysisData().NumBlockExprs);
}

//===--------------------------------------------------------------------===//
// Transfer functions.
//===--------------------------------------------------------------------===//      

namespace {

class TransferFuncs : public CFGStmtVisitor<TransferFuncs,bool> {
  UninitializedValues::ValTy V;
  UninitializedValues::AnalysisDataTy& AD;
public:
  TransferFuncs(UninitializedValues::AnalysisDataTy& ad) : AD(ad) {
    V.DeclBV.resize(AD.NumDecls);
    V.ExprBV.resize(AD.NumBlockExprs);
  }
  
  UninitializedValues::ValTy& getVal() { return V; }
  
  bool VisitDeclRefExpr(DeclRefExpr* DR);
  bool VisitBinaryOperator(BinaryOperator* B);
  bool VisitUnaryOperator(UnaryOperator* U);
  bool VisitStmt(Stmt* S);
  bool VisitCallExpr(CallExpr* C);
  bool BlockStmt_VisitExpr(Expr* E);
  
  static inline bool Initialized() { return true; }
  static inline bool Unintialized() { return false; }
};


bool TransferFuncs::VisitDeclRefExpr(DeclRefExpr* DR) {
  if (VarDecl* VD = dyn_cast<VarDecl>(DR->getDecl())) {
    assert ( AD.VMap.find(VD) != AD.VMap.end() && "Unknown VarDecl.");
    return V.DeclBV[ AD.VMap[VD] ];    
  }
  else
    return Initialized();
}

bool TransferFuncs::VisitBinaryOperator(BinaryOperator* B) {
  if (CFG::hasImplicitControlFlow(B)) {
    assert ( AD.EMap.find(B) != AD.EMap.end() && "Unknown block-level expr.");
    return V.ExprBV[ AD.EMap[B] ];
  }

  return VisitStmt(B);
}

bool TransferFuncs::VisitCallExpr(CallExpr* C) {
  VisitStmt(C);
  return Initialized();
}

bool TransferFuncs::VisitUnaryOperator(UnaryOperator* U) {
  switch (U->getOpcode()) {
    case UnaryOperator::AddrOf: {
      // Blast through parentheses and find the decl (if any).  Treat it
      // as initialized from this point forward.
      for (Stmt* S = U->getSubExpr() ;; )
        if (ParenExpr* P = dyn_cast<ParenExpr>(S))
          S = P->getSubExpr();
        else if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(S)) {
          if (VarDecl* VD = dyn_cast<VarDecl>(DR->getDecl())) {
            assert ( AD.VMap.find(VD) != AD.VMap.end() && "Unknown VarDecl.");
            V.DeclBV[ AD.VMap[VD] ] = Initialized();
          }
          break;
        }
        else {
          // Evaluate the transfer function for subexpressions, even
          // if we cannot reason more deeply about the &-expression.
          return Visit(U->getSubExpr());
        }

      return Initialized();
    }

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
    if (Visit(*I) == Unintialized())
      x = Unintialized();
  
  return x;
}

bool TransferFuncs::BlockStmt_VisitExpr(Expr* E) {
  assert ( AD.EMap.find(E) != AD.EMap.end() );
  return V.ExprBV[ AD.EMap[E] ] = Visit(E);
}
  
} // end anonymous namespace

//===--------------------------------------------------------------------===//
// Merge operator.
//
//  In our transfer functions we take the approach that any
//  combination of unintialized values, e.g. Unitialized + ___ = Unitialized.
//
//  Merges take the opposite approach.
//
//  In the merge of dataflow values (for Decls) we prefer unsoundness, and
//  prefer false negatives to false positives.  At merges, if a value for a
//  tracked Decl is EVER initialized in any of the predecessors we treat it as
//  initialized at the confluence point.
//
//  For tracked CFGBlock-level expressions (such as the result of
//  short-circuit), we do the opposite merge: if a value is EVER uninitialized
//  in a predecessor we treat it as uninitalized at the confluence point.
//  The reason we do this is because dataflow values for tracked Exprs are
//  not as control-dependent as dataflow values for tracked Decls.
//===--------------------------------------------------------------------===//      

namespace {
struct Merge {
  void operator()(UninitializedValues::ValTy& Dst,
                  UninitializedValues::ValTy& Src) {
    assert (Dst.DeclBV.size() == Src.DeclBV.size() 
            && "Bitvector sizes do not match.");
            
    Dst.DeclBV |= Src.DeclBV;
    
    assert (Dst.ExprBV.size() == Src.ExprBV.size()
            && "Bitvector sizes do not match.");

    Dst.ExprBV &= Src.ExprBV;
  }
};
} // end anonymous namespace

//===--------------------------------------------------------------------===//
// External interface (driver logic).
//===--------------------------------------------------------------------===//      

void UninitializedValues::CheckUninitializedValues(const CFG& cfg) {

  typedef DataflowSolver<UninitializedValues,TransferFuncs,Merge> Solver;

  UninitializedValues U;
  
  { // Compute the unitialized values information.
    Solver S(U);
    S.runOnCFG(cfg);
  }

//  WarnObserver O;
  Solver S(U);
    
  for (CFG::const_iterator I=cfg.begin(), E=cfg.end(); I!=E; ++I)
    S.runOnBlock(&*I);
}
