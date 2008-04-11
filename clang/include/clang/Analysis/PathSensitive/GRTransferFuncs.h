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
#include "clang/Analysis/PathSensitive/ValueState.h"

namespace clang {
  
  class GRExprEngine;
  
class GRTransferFuncs {
public:
  GRTransferFuncs() {}
  virtual ~GRTransferFuncs() {}
  
  virtual ValueState::CheckerStatePrinter* getCheckerStatePrinter() {
    return NULL;
  }
  
  virtual void RegisterChecks(GRExprEngine& Eng) {}
  
  // Casts.
  
  virtual RVal EvalCast(GRExprEngine& Engine, NonLVal V, QualType CastT) =0;  
  virtual RVal EvalCast(GRExprEngine& Engine, LVal V, QualType CastT) = 0;

  // Unary Operators.
  
  virtual RVal EvalMinus(GRExprEngine& Engine, UnaryOperator* U, NonLVal X) = 0;

  virtual RVal EvalComplement(GRExprEngine& Engine, NonLVal X) = 0;

  // Binary Operators.
  
  virtual RVal EvalBinOp(GRExprEngine& Engine, BinaryOperator::Opcode Op,
                         NonLVal L, NonLVal R) = 0;
  
  virtual RVal EvalBinOp(GRExprEngine& Engine, BinaryOperator::Opcode Op,
                         LVal L, LVal R) = 0;
  
  // Pointer arithmetic.
  
  virtual RVal EvalBinOp(GRExprEngine& Engine, BinaryOperator::Opcode Op,
                         LVal L, NonLVal R) = 0;
  
  // Calls.
  
  virtual void EvalCall(ExplodedNodeSet<ValueState>& Dst,
                        GRExprEngine& Engine,
                        GRStmtNodeBuilder<ValueState>& Builder,
                        CallExpr* CE, LVal L,
                        ExplodedNode<ValueState>* Pred) = 0;
  
  // End-of-path.
  
  virtual void EvalEndPath(GRExprEngine& Engine,
                           GREndPathNodeBuilder<ValueState>& Builder) {}
};
  
} // end clang namespace

#endif
