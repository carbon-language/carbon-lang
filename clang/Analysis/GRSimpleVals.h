// GRSimpleVals.h - Transfer functions for tracking simple values -*- C++ -*--//
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

#ifndef LLVM_CLANG_ANALYSIS_GRSIMPLEVALS
#define LLVM_CLANG_ANALYSIS_GRSIMPLEVALS

#include "clang/Analysis/PathSensitive/GRTransferFuncs.h"

namespace clang {
  
class GRSimpleVals : public GRTransferFuncs {
public:
  GRSimpleVals() {}
  virtual ~GRSimpleVals() {}
  
  virtual RValue EvalCast(ValueManager& ValMgr, NonLValue V, Expr* CastExpr);
  virtual RValue EvalCast(ValueManager& ValMgr, LValue V, Expr* CastExpr);
};
  
  
} // end clang namespace

#endif
