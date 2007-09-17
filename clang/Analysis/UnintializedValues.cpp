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
      AD.VMap[D] = AD.Counter++;
  }
  
  void BlockStmt_VisitExpr(Expr* E) {
    if (AD.EMap.find(E) == AD.EMap.end())
      AD.EMap[E] = AD.Counter++;
  }        
};
  
} // end anonymous namespace

void UninitializedValues::InitializeValues(const CFG& cfg) {
  RegisterDeclsAndExprs R(cfg,this->getAnalysisData());
  R.VisitAllDecls();    
  getBlockDataMap()[ &cfg.getEntry() ].resize( getAnalysisData().Counter );
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
    V.resize(AD.Counter);
  }
  
  UninitializedValues::ValTy& getVal() { return V; }
  
//  bool VisitDeclRefExpr(DeclRefExpr* DR);
//  bool VisitBinaryOperator(BinaryOperator* B);
//  bool VisitUnaryOperator(UnaryOperator* U);    
};
} // end anonymous namespace

//===--------------------------------------------------------------------===//
// Merge operator.
//===--------------------------------------------------------------------===//      

namespace {
struct Merge {
  void operator()(UninitializedValues::ValTy& Dst,
                  UninitializedValues::ValTy& Src) {
    assert (Dst.size() == Src.size() && "Bitvector sizes do not match.");
    Src |= Dst;
  }
};
} // end anonymous namespace

//===--------------------------------------------------------------------===//
// Observer to flag warnings for uses of uninitialized variables.
//===--------------------------------------------------------------------===//      




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
