//== GRTransferFuncs.h - Path-Sens. Transfer Functions Interface -*- C++ -*--=//
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

#ifndef LLVM_CLANG_ANALYSIS_GRTF
#define LLVM_CLANG_ANALYSIS_GRTF

#include "clang/Analysis/PathSensitive/RValues.h"

namespace clang {
  
class GRTransferFuncs {
public:
  GRTransferFuncs() {}
  virtual ~GRTransferFuncs() {}
  
  RValue EvalCast(ValueManager& ValMgr, RValue V, Expr* CastExpr);
  virtual RValue EvalCast(ValueManager& ValMgr, NonLValue V, Expr* CastExpr) =0;
  virtual RValue EvalCast(ValueManager& ValMgr, LValue V, Expr* CastExpr) = 0;

  
  
};
  
} // end clang namespace

#endif
