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
  class ObjCMessageExpr;
  
class GRTransferFuncs {
public:
  GRTransferFuncs() {}
  virtual ~GRTransferFuncs() {}
  
  virtual ValueState::CheckerStatePrinter* getCheckerStatePrinter() {
    return NULL;
  }
  
  virtual void RegisterChecks(GRExprEngine& Eng);
  
  // Casts.
  
  virtual RVal EvalCast(GRExprEngine& Engine, NonLVal V, QualType CastT) =0;  
  virtual RVal EvalCast(GRExprEngine& Engine, LVal V, QualType CastT) = 0;

  // Unary Operators.
  
  virtual RVal EvalMinus(GRExprEngine& Engine, UnaryOperator* U, NonLVal X) = 0;

  virtual RVal EvalComplement(GRExprEngine& Engine, NonLVal X) = 0;

  // Binary Operators.
  
  virtual RVal EvalBinOp(ValueStateManager& StateMgr, BinaryOperator::Opcode Op,
                         NonLVal L, NonLVal R) {
    return UnknownVal();
  }
  
  virtual void EvalBinOpNN(ValueStateSet& OStates, ValueStateManager& StateMgr,
                           const ValueState* St, Expr* Ex,
                           BinaryOperator::Opcode Op, NonLVal L, NonLVal R);
  
  virtual RVal EvalBinOp(GRExprEngine& Engine, BinaryOperator::Opcode Op,
                         LVal L, LVal R) = 0;
  
  // Pointer arithmetic.
  
  virtual RVal EvalBinOp(GRExprEngine& Engine, BinaryOperator::Opcode Op,
                         LVal L, NonLVal R) = 0;
  
  // Calls.
  
  virtual void EvalCall(ExplodedNodeSet<ValueState>& Dst,
                        GRExprEngine& Engine,
                        GRStmtNodeBuilder<ValueState>& Builder,
                        CallExpr* CE, RVal L,
                        ExplodedNode<ValueState>* Pred) {}
  
  virtual void EvalObjCMessageExpr(ExplodedNodeSet<ValueState>& Dst,
                                   GRExprEngine& Engine,
                                   GRStmtNodeBuilder<ValueState>& Builder,
                                   ObjCMessageExpr* ME,
                                   ExplodedNode<ValueState>* Pred) {}
  
  // Stores.
  
  /// EvalStore - Evaluate the effects of a store, creating a new node
  ///  the represents the effect of binding 'Val' to the location 'TargetLV'.
  //   TargetLV is guaranteed to either be an UnknownVal or an LVal.
  virtual void EvalStore(ExplodedNodeSet<ValueState>& Dst,
                         GRExprEngine& Engine,
                         GRStmtNodeBuilder<ValueState>& Builder,
                         Expr* E, ExplodedNode<ValueState>* Pred,
                         const ValueState* St, RVal TargetLV, RVal Val);
                         
  
  // End-of-path and dead symbol notification.
  
  virtual void EvalEndPath(GRExprEngine& Engine,
                           GREndPathNodeBuilder<ValueState>& Builder) {}
  
  
  virtual void EvalDeadSymbols(ExplodedNodeSet<ValueState>& Dst,
                               GRExprEngine& Engine,
                               GRStmtNodeBuilder<ValueState>& Builder,
                               ExplodedNode<ValueState>* Pred,
                               Stmt* S,
                               const ValueState* St,
                               const ValueStateManager::DeadSymbolsTy& Dead) {}
  
  // Return statements.  
  virtual void EvalReturn(ExplodedNodeSet<ValueState>& Dst,
                          GRExprEngine& Engine,
                          GRStmtNodeBuilder<ValueState>& Builder,
                          ReturnStmt* S,
                          ExplodedNode<ValueState>* Pred) {}

  // Assumptions.
  
  virtual const ValueState* EvalAssume(ValueStateManager& VMgr,
                                       const ValueState* St,
                                       RVal Cond, bool Assumption,
                                       bool& isFeasible) {
    return St;
  }
};
  
} // end clang namespace

#endif
