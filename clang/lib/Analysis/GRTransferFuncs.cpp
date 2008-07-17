//== GRTransferFuncs.cpp - Path-Sens. Transfer Functions Interface -*- C++ -*--=
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines GRTransferFuncs, which provides a base-class that
//  defines an interface for transfer functions used by GRExprEngine.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/GRTransferFuncs.h"
#include "clang/Analysis/PathSensitive/GRExprEngine.h"

using namespace clang;

void GRTransferFuncs::RegisterChecks(GRExprEngine& Eng) {}

void GRTransferFuncs::EvalStore(ExplodedNodeSet<ValueState>& Dst,
                                GRExprEngine& Eng,
                                GRStmtNodeBuilder<ValueState>& Builder,
                                Expr* E, ExplodedNode<ValueState>* Pred,
                                const ValueState* St, RVal TargetLV, RVal Val) {
  
  // This code basically matches the "safety-net" logic of GRExprEngine:
  //  bind Val to TargetLV, and create a new node.  We replicate it here
  //  because subclasses of GRTransferFuncs may wish to call it.

  assert (!TargetLV.isUndef());
  
  if (TargetLV.isUnknown())
    Builder.MakeNode(Dst, E, Pred, St);
  else
    Builder.MakeNode(Dst, E, Pred,
                Eng.getStateManager().SetRVal(St, cast<LVal>(TargetLV), Val));    
}

void GRTransferFuncs::EvalBinOpNN(ExplodedNodeSet<ValueState>& Dst,
                                  GRExprEngine& Engine,
                                  GRStmtNodeBuilder<ValueState>& Builder,
                                  BinaryOperator::Opcode Op,
                                  Expr* Ex,
                                  NonLVal L, NonLVal R,
                                  ExplodedNode<ValueState>* Pred) {

  ValueStateManager& StateMgr = Engine.getStateManager();
  const ValueState* St = Builder.GetState(Pred);
  
  RVal Result = EvalBinOp(Engine, Op, L, R);
  Builder.MakeNode(Dst, Ex, Pred, StateMgr.SetRVal(St, Ex, Result));
}
