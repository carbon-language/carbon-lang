// CFRefCount.cpp - Transfer functions for tracking simple values -*- C++ -*--
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the methods for CFRefCount, which implements
//  a reference count checker for Core Foundation (Mac OS X).
//
//===----------------------------------------------------------------------===//

#include "CFRefCount.h"
#include "clang/Analysis/PathSensitive/ValueState.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Analysis/LocalCheckers.h"


using namespace clang;


namespace clang {
  
void CheckCFRefCount(CFG& cfg, FunctionDecl& FD, ASTContext& Ctx,
                     Diagnostic& Diag) {
  
  if (Diag.hasErrorOccurred())
    return;
  
  // FIXME: Refactor some day so this becomes a single function invocation.
  
  GRCoreEngine<GRExprEngine> Engine(cfg, FD, Ctx);
  GRExprEngine* CS = &Engine.getCheckerState();
  CFRefCount TF;
  CS->setTransferFunctions(TF);
  Engine.ExecuteWorkList(20000);
  
}
  
}

void CFRefCount::EvalCall(ExplodedNodeSet<ValueState>& Dst,
                            ValueStateManager& StateMgr,
                            GRStmtNodeBuilder<ValueState>& Builder,
                            BasicValueFactory& BasicVals,
                            CallExpr* CE, LVal L,
                            ExplodedNode<ValueState>* Pred) {
  
  ValueState* St = Pred->getState();
  
  // Invalidate all arguments passed in by reference (LVals).

  for (CallExpr::arg_iterator I = CE->arg_begin(), E = CE->arg_end();
        I != E; ++I) {

    RVal V = StateMgr.GetRVal(St, *I);
    
    if (isa<LVal>(V))
      St = StateMgr.SetRVal(St, cast<LVal>(V), UnknownVal());
  }
    
  Builder.Nodify(Dst, CE, Pred, St);
}
