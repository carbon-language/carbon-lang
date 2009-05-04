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
protected:
  
  virtual SVal DetermEvalBinOpNN(GRExprEngine& Eng,
                                 BinaryOperator::Opcode Op,
                                 NonLoc L, NonLoc R, QualType T);
  
public:
  GRSimpleVals() {}
  virtual ~GRSimpleVals() {}
  
  // Casts.
  
  virtual SVal EvalCast(GRExprEngine& Engine, NonLoc V, QualType CastT);
  virtual SVal EvalCast(GRExprEngine& Engine, Loc V, QualType CastT);
  
  // Unary Operators.
  
  virtual SVal EvalMinus(GRExprEngine& Engine, UnaryOperator* U, NonLoc X);

  virtual SVal EvalComplement(GRExprEngine& Engine, NonLoc X);
  
  // Binary Operators.
  
  virtual SVal EvalBinOp(GRExprEngine& Engine, BinaryOperator::Opcode Op,
                         Loc L, Loc R);
  
  // Pointer arithmetic.
  
  virtual SVal EvalBinOp(GRExprEngine& Engine, BinaryOperator::Opcode Op,
                         Loc L, NonLoc R);  
  
  // Calls.
  
  virtual void EvalCall(ExplodedNodeSet<GRState>& Dst,
                        GRExprEngine& Engine,
                        GRStmtNodeBuilder<GRState>& Builder,
                        CallExpr* CE, SVal L,
                        ExplodedNode<GRState>* Pred);
  
  virtual void EvalObjCMessageExpr(ExplodedNodeSet<GRState>& Dst,
                                   GRExprEngine& Engine,
                                   GRStmtNodeBuilder<GRState>& Builder,
                                   ObjCMessageExpr* ME,
                                   ExplodedNode<GRState>* Pred);
  
  
  
  static void GeneratePathDiagnostic(PathDiagnostic& PD, ASTContext& Ctx,
                                     ExplodedNode<GRState>* N);
  
protected:
  
  // Equality (==, !=) operators for Locs.  
  SVal EvalEquality(GRExprEngine& Engine, Loc L, Loc R, bool isEqual);
};
  
} // end clang namespace

#endif
