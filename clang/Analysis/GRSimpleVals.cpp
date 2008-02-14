// GRSimpleVals.cpp - Transfer functions for tracking simple values -*- C++ -*--
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This files defines GRSimpleVals, a sub-class of GRTransferFuncs that
//  provides transfer functions for performing simple value tracking with
//  limited support for symbolics.
//
//===----------------------------------------------------------------------===//

#include "GRSimpleVals.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// Transfer function for Casts.
//===----------------------------------------------------------------------===//

RValue GRSimpleVals::EvalCast(ValueManager& ValMgr, NonLValue X,
                              Expr* CastExpr) {
  
  if (!isa<nonlval::ConcreteInt>(X))
    return UnknownVal();
  
  llvm::APSInt V = cast<nonlval::ConcreteInt>(X).getValue();
  QualType T = CastExpr->getType();
  V.setIsUnsigned(T->isUnsignedIntegerType() || T->isPointerType());
  V.extOrTrunc(ValMgr.getContext().getTypeSize(T, CastExpr->getLocStart()));
  
  if (CastExpr->getType()->isPointerType())
    return lval::ConcreteInt(ValMgr.getValue(V));
  else
    return nonlval::ConcreteInt(ValMgr.getValue(V));
}

// Casts.

RValue GRSimpleVals::EvalCast(ValueManager& ValMgr, LValue X, Expr* CastExpr) {

  if (CastExpr->getType()->isPointerType())
    return X;
  
  assert (CastExpr->getType()->isIntegerType());
  
  if (!isa<lval::ConcreteInt>(X))
    return UnknownVal();
  
  llvm::APSInt V = cast<lval::ConcreteInt>(X).getValue();
  QualType T = CastExpr->getType();
  V.setIsUnsigned(T->isUnsignedIntegerType() || T->isPointerType());
  V.extOrTrunc(ValMgr.getContext().getTypeSize(T, CastExpr->getLocStart()));

  return nonlval::ConcreteInt(ValMgr.getValue(V));
}

// Unary operators.

NonLValue GRSimpleVals::EvalMinus(ValueManager& ValMgr, UnaryOperator* U,
                                  NonLValue X) {
  
  switch (X.getSubKind()) {
    case nonlval::ConcreteIntKind:
      return cast<nonlval::ConcreteInt>(X).EvalMinus(ValMgr, U);
    default:
      return cast<NonLValue>(UnknownVal());
  }
}

NonLValue GRSimpleVals::EvalComplement(ValueManager& ValMgr, NonLValue X) {
  switch (X.getSubKind()) {
    case nonlval::ConcreteIntKind:
      return cast<nonlval::ConcreteInt>(X).EvalComplement(ValMgr);
    default:
      return cast<NonLValue>(UnknownVal());
  }
}
