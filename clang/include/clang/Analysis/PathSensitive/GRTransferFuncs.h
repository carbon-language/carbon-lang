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
#include "clang/Analysis/PathSensitive/GRCoreEngine.h"

namespace clang {
  
  class ValueState;
  class ValueStateManager;
  
class GRTransferFuncs {
public:
  GRTransferFuncs() {}
  virtual ~GRTransferFuncs() {}
  
  // Casts.
  
  virtual RVal EvalCast(ValueManager& ValMgr, NonLVal V, QualType CastT) =0;
  virtual RVal EvalCast(ValueManager& ValMgr, LVal V, QualType CastT) = 0;

  // Unary Operators.
  
  virtual RVal EvalMinus(ValueManager& ValMgr, UnaryOperator* U, NonLVal X) = 0;

  virtual RVal EvalComplement(ValueManager& ValMgr, NonLVal X) = 0;

  // Binary Operators.
  
  virtual RVal EvalBinOp(ValueManager& ValMgr, BinaryOperator::Opcode Op,
                         NonLVal L, NonLVal R) = 0;
  
  virtual RVal EvalBinOp(ValueManager& ValMgr, BinaryOperator::Opcode Op,
                         LVal L, LVal R) = 0;
  
  // Pointer arithmetic.
  
  virtual RVal EvalBinOp(ValueManager& ValMgr, BinaryOperator::Opcode Op,
                         LVal L, NonLVal R) = 0;
  
  // Calls.
  
  virtual void EvalCall(ExplodedNodeSet<ValueState>& Dst,
                        ValueStateManager& StateMgr,
                        GRStmtNodeBuilder<ValueState>& Builder,
                        ValueManager& ValMgr, CallExpr* CE, LVal L,
                        ExplodedNode<ValueState>* Pred) = 0;
};
  
} // end clang namespace

#endif
