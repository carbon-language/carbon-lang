//== GRTransferFuncs.cpp - Path-Sens. Transfer Functions Interface -*- C++ -*--=
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This files defines GRTransferFuncs, which provides a base-class that
//  defines an interface for transfer functions used by GRExprEngine.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/GRTransferFuncs.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// Transfer function for Casts.
//===----------------------------------------------------------------------===//

RValue GRTransferFuncs::EvalCast(ValueManager& ValMgr, RValue X,
                                 Expr* CastExpr) {
  
  switch (X.getBaseKind()) {
    default:
      assert(false && "Invalid RValue."); break;

    case RValue::LValueKind: 
      return EvalCast(ValMgr, cast<LValue>(X), CastExpr);

    case RValue::NonLValueKind:
      return EvalCast(ValMgr, cast<NonLValue>(X), CastExpr);
    
    case RValue::UninitializedKind:
    case RValue::UnknownKind: break;
  }
  
  return X;
}

// Binary Operators (except assignments and comma).

RValue GRTransferFuncs::EvalBinaryOp(ValueManager& ValMgr,
                                     BinaryOperator::Opcode Op,
                                     LValue LHS, LValue RHS) {
  
  switch (Op) {
    default:
      assert (false && "Not yet implemented.");
      
    case BinaryOperator::EQ:
      return EvalEQ(ValMgr, LHS, RHS);
      
    case BinaryOperator::NE:
      return EvalNE(ValMgr, LHS, RHS);
  }
}