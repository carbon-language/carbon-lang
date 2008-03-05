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
#include "clang/Analysis/PathSensitive/GRExprEngine.h"

namespace clang {
  
class GRSimpleVals : public GRTransferFuncs {
public:
  GRSimpleVals() {}
  virtual ~GRSimpleVals() {}
  
  // Casts.
  
  virtual RVal EvalCast(ValueManager& ValMgr, NonLVal V, QualType CastT);
  virtual RVal EvalCast(ValueManager& ValMgr, LVal V, QualType CastT);
  
  // Unary Operators.
  
  virtual RVal EvalMinus(ValueManager& ValMgr, UnaryOperator* U, NonLVal X);

  virtual RVal EvalComplement(ValueManager& ValMgr, NonLVal X);
  
  // Binary Operators.
  
  virtual RVal EvalBinOp(ValueManager& ValMgr, BinaryOperator::Opcode Op,
                         NonLVal L, NonLVal R);
  
  virtual RVal EvalBinOp(ValueManager& ValMgr, BinaryOperator::Opcode Op,
                         LVal L, LVal R);
  
  // Pointer arithmetic.
  
  virtual RVal EvalBinOp(ValueManager& ValMgr, BinaryOperator::Opcode Op,
                         LVal L, NonLVal R);  
  
  // Calls.
  
  virtual void EvalCall(ExplodedNodeSet<ValueState>& Dst,
                        ValueStateManager& StateMgr,
                        GRStmtNodeBuilder<ValueState>& Builder,
                        ValueManager& ValMgr,
                        CallExpr* CE, LVal L,
                        ExplodedNode<ValueState>* Pred);
  
protected:
  
  // Equality operators for LVals.
  
  RVal EvalEQ(ValueManager& ValMgr, LVal L, LVal R);
  RVal EvalNE(ValueManager& ValMgr, LVal L, LVal R);
};
  
} // end clang namespace

#endif
