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

void GRTransferFuncs::EvalBinOpNN(GRStateSet& OStates,
                                  GRExprEngine& Eng,
                                  const GRState *St, Expr* Ex,
                                  BinaryOperator::Opcode Op,
                                  NonLoc L, NonLoc R, QualType T) {
  
  OStates.Add(Eng.getStateManager().BindExpr(St, Ex,
                                          DetermEvalBinOpNN(Eng, Op, L, R, T)));
}
