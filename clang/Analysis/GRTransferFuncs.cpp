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

RVal GRTransferFuncs::EvalCast(ValueManager& ValMgr, RVal X, Expr* CastExpr) {
  
  switch (X.getBaseKind()) {
      
    default:
      assert(false && "Invalid RVal."); break;

    case RVal::LValKind: 
      return EvalCast(ValMgr, cast<LVal>(X), CastExpr);

    case RVal::NonLValKind:
      return EvalCast(ValMgr, cast<NonLVal>(X), CastExpr);
    
    case RVal::UninitializedKind:
    case RVal::UnknownKind: break;
  }
  
  return X;
}
