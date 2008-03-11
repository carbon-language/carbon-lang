//== GRTransferFuncs.h - Path-Sens. Transfer Functions Interface -*- C++ -*--=//
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
  
  virtual RVal EvalCast(BasicValueFactory& BasicVals, NonLVal V,
                        QualType CastT) =0;
  
  virtual RVal EvalCast(BasicValueFactory& BasicVals, LVal V,
                        QualType CastT) = 0;

  // Unary Operators.
  
  virtual RVal EvalMinus(BasicValueFactory& BasicVals, UnaryOperator* U,
                         NonLVal X) = 0;

  virtual RVal EvalComplement(BasicValueFactory& BasicVals, NonLVal X) = 0;

  // Binary Operators.
  
  virtual RVal EvalBinOp(BasicValueFactory& BasicVals,
                         BinaryOperator::Opcode Op, NonLVal L, NonLVal R) = 0;
  
  virtual RVal EvalBinOp(BasicValueFactory& BasicVals,
                         BinaryOperator::Opcode Op, LVal L, LVal R) = 0;
  
  // Pointer arithmetic.
  
  virtual RVal EvalBinOp(BasicValueFactory& BasicVals,
                         BinaryOperator::Opcode Op, LVal L, NonLVal R) = 0;
  
  // Calls.
  
  virtual void EvalCall(ExplodedNodeSet<ValueState>& Dst,
                        ValueStateManager& StateMgr,
                        GRStmtNodeBuilder<ValueState>& Builder,
                        BasicValueFactory& BasicVals, CallExpr* CE, LVal L,
                        ExplodedNode<ValueState>* Pred) = 0;
};
  
} // end clang namespace

#endif
