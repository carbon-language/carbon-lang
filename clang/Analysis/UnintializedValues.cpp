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

class RegisterDecls : public CFGVarDeclVisitor<RegisterDecls> {
  UninitializedValues::MetaDataTy& M;
public:
  RegisterDecls(const CFG& cfg, UninitializedValues::MetaDataTy& m) :
    CFGVarDeclVisitor<RegisterDecls>(cfg), M(m) {}
  
  void VisitVarDecl(VarDecl* D) {
    if (M.Map.find(D) == M.Map.end()) {
      M.Map[D] = M.NumDecls++;
    }
  }  
};
  
} // end anonymous namespace

void UninitializedValues::InitializeValues(const CFG& cfg) {
  RegisterDecls R(cfg,this->getMetaData());
  R.VisitAllDecls();
    
  getBlockDataMap()[ &cfg.getEntry() ].resize( getMetaData().NumDecls );
}

//===--------------------------------------------------------------------===//
// Transfer functions.
//===--------------------------------------------------------------------===//      

namespace {
class TransferFuncs : public CFGStmtVisitor<TransferFuncs,bool> {
  UninitializedValues::ValTy V;
  UninitializedValues::MetaDataTy& M;
  UninitializedValues::ObserverTy* O;
public:
  TransferFuncs(UninitializedValues::MetaDataTy& m,
                UninitializedValues::ObserverTy* o) : M(m), O(o) {
    V.resize(M.NumDecls);
  }
  
  UninitializedValues::ValTy& getVal() { return V; }
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
