// GRSimpleVals.h - Transfer functions for tracking simple values -*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines GRSimpleVals, a sub-class of GRTransferFuncs that
//  provides transfer functions for performing simple value tracking with
//  limited support for symbolics.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_GRSIMPLEVALS
#define LLVM_CLANG_ANALYSIS_GRSIMPLEVALS

#include "clang/Analysis/PathSensitive/GRTransferFuncs.h"
#include "clang/Analysis/PathSensitive/GRExprEngine.h"

namespace clang {
  
class PathDiagnostic;
class ASTContext;
  
class GRSimpleVals : public GRTransferFuncs {
public:
  GRSimpleVals() {}
  virtual ~GRSimpleVals() {}
  
  virtual void RegisterChecks(GRExprEngine& Eng);
  
  // Casts.
  
  virtual RVal EvalCast(GRExprEngine& Engine, NonLVal V, QualType CastT);
  virtual RVal EvalCast(GRExprEngine& Engine, LVal V, QualType CastT);
  
  // Unary Operators.
  
  virtual RVal EvalMinus(GRExprEngine& Engine, UnaryOperator* U, NonLVal X);

  virtual RVal EvalComplement(GRExprEngine& Engine, NonLVal X);
  
  // Binary Operators.
  
  virtual RVal EvalBinOp(ValueStateManager& StateMgr, BinaryOperator::Opcode Op,
                         NonLVal L, NonLVal R);
  
  virtual RVal EvalBinOp(GRExprEngine& Engine, BinaryOperator::Opcode Op,
                         LVal L, LVal R);
  
  // Pointer arithmetic.
  
  virtual RVal EvalBinOp(GRExprEngine& Engine, BinaryOperator::Opcode Op,
                         LVal L, NonLVal R);  
  
  // Calls.
  
  virtual void EvalCall(ExplodedNodeSet<ValueState>& Dst,
                        GRExprEngine& Engine,
                        GRStmtNodeBuilder<ValueState>& Builder,
                        CallExpr* CE, RVal L,
                        ExplodedNode<ValueState>* Pred);
  
  virtual void EvalObjCMessageExpr(ExplodedNodeSet<ValueState>& Dst,
                                   GRExprEngine& Engine,
                                   GRStmtNodeBuilder<ValueState>& Builder,
                                   ObjCMessageExpr* ME,
                                   ExplodedNode<ValueState>* Pred);
  
  
  
  static void GeneratePathDiagnostic(PathDiagnostic& PD, ASTContext& Ctx,
                                     ExplodedNode<ValueState>* N);
  
protected:
  
  // Equality operators for LVals.
  
  RVal EvalEQ(GRExprEngine& Engine, LVal L, LVal R);
  RVal EvalNE(GRExprEngine& Engine, LVal L, LVal R);
};
  
} // end clang namespace

#endif
