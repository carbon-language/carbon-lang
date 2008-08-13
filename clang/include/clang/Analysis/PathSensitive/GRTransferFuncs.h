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
#include "clang/Analysis/PathSensitive/GRState.h"

namespace clang {
  
  class GRExprEngine;
  class ObjCMessageExpr;
  
class GRTransferFuncs {
  
  friend class GRExprEngine;
  
protected:
  
  
  virtual RVal DetermEvalBinOpNN(GRStateManager& StateMgr,
                                 BinaryOperator::Opcode Op,
                                 NonLVal L, NonLVal R) {
    return UnknownVal();
  }
  
  
public:
  GRTransferFuncs() {}
  virtual ~GRTransferFuncs() {}
  
  virtual GRState::CheckerStatePrinter* getCheckerStatePrinter() {
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
  virtual void EvalBinOpNN(GRStateSet& OStates, GRStateManager& StateMgr,
                           const GRState* St, Expr* Ex,
                           BinaryOperator::Opcode Op, NonLVal L, NonLVal R);
  
  virtual RVal EvalBinOp(GRExprEngine& Engine, BinaryOperator::Opcode Op,
                         LVal L, LVal R) = 0;
  
  // Pointer arithmetic.
  
  virtual RVal EvalBinOp(GRExprEngine& Engine, BinaryOperator::Opcode Op,
                         LVal L, NonLVal R) = 0;
  
  // Calls.
  
  virtual void EvalCall(ExplodedNodeSet<GRState>& Dst,
                        GRExprEngine& Engine,
                        GRStmtNodeBuilder<GRState>& Builder,
                        CallExpr* CE, RVal L,
                        ExplodedNode<GRState>* Pred) {}
  
  virtual void EvalObjCMessageExpr(ExplodedNodeSet<GRState>& Dst,
                                   GRExprEngine& Engine,
                                   GRStmtNodeBuilder<GRState>& Builder,
                                   ObjCMessageExpr* ME,
                                   ExplodedNode<GRState>* Pred) {}
  
  // Stores.
  
  /// EvalStore - Evaluate the effects of a store, creating a new node
  ///  the represents the effect of binding 'Val' to the location 'TargetLV'.
  //   TargetLV is guaranteed to either be an UnknownVal or an LVal.
  virtual void EvalStore(ExplodedNodeSet<GRState>& Dst,
                         GRExprEngine& Engine,
                         GRStmtNodeBuilder<GRState>& Builder,
                         Expr* E, ExplodedNode<GRState>* Pred,
                         const GRState* St, RVal TargetLV, RVal Val);
                         
  
  // End-of-path and dead symbol notification.
  
  virtual void EvalEndPath(GRExprEngine& Engine,
                           GREndPathNodeBuilder<GRState>& Builder) {}
  
  
  virtual void EvalDeadSymbols(ExplodedNodeSet<GRState>& Dst,
                               GRExprEngine& Engine,
                               GRStmtNodeBuilder<GRState>& Builder,
                               ExplodedNode<GRState>* Pred,
                               Stmt* S,
                               const GRState* St,
                               const GRStateManager::DeadSymbolsTy& Dead) {}
  
  // Return statements.  
  virtual void EvalReturn(ExplodedNodeSet<GRState>& Dst,
                          GRExprEngine& Engine,
                          GRStmtNodeBuilder<GRState>& Builder,
                          ReturnStmt* S,
                          ExplodedNode<GRState>* Pred) {}

  // Assumptions.
  
  virtual const GRState* EvalAssume(GRStateManager& VMgr,
                                       const GRState* St,
                                       RVal Cond, bool Assumption,
                                       bool& isFeasible) {
    return St;
  }
};
  
} // end clang namespace

#endif
