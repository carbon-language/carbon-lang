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
#include <vector>

namespace clang {
  
  class GRExprEngine;
  class ObjCMessageExpr;
  
class GRTransferFuncs {
  
  friend class GRExprEngine;
  
protected:
  
  
  virtual SVal DetermEvalBinOpNN(GRStateManager& StateMgr,
                                 BinaryOperator::Opcode Op,
                                 NonLoc L, NonLoc R) {
    return UnknownVal();
  }
  
  
public:
  GRTransferFuncs() {}
  virtual ~GRTransferFuncs() {}
  
  virtual void RegisterPrinters(std::vector<GRState::Printer*>& Printers) {}
  virtual void RegisterChecks(GRExprEngine& Eng);
  
  // Casts.
  
  virtual SVal EvalCast(GRExprEngine& Engine, NonLoc V, QualType CastT) =0;  
  virtual SVal EvalCast(GRExprEngine& Engine, Loc V, QualType CastT) = 0;

  // Unary Operators.
  
  virtual SVal EvalMinus(GRExprEngine& Engine, UnaryOperator* U, NonLoc X) = 0;

  virtual SVal EvalComplement(GRExprEngine& Engine, NonLoc X) = 0;

  // Binary Operators.
  virtual void EvalBinOpNN(GRStateSet& OStates, GRStateManager& StateMgr,
                           const GRState* St, Expr* Ex,
                           BinaryOperator::Opcode Op, NonLoc L, NonLoc R);
  
  virtual SVal EvalBinOp(GRExprEngine& Engine, BinaryOperator::Opcode Op,
                         Loc L, Loc R) = 0;
  
  // Pointer arithmetic.
  
  virtual SVal EvalBinOp(GRExprEngine& Engine, BinaryOperator::Opcode Op,
                         Loc L, NonLoc R) = 0;
  
  // Calls.
  
  virtual void EvalCall(ExplodedNodeSet<GRState>& Dst,
                        GRExprEngine& Engine,
                        GRStmtNodeBuilder<GRState>& Builder,
                        CallExpr* CE, SVal L,
                        ExplodedNode<GRState>* Pred) {}
  
  virtual void EvalObjCMessageExpr(ExplodedNodeSet<GRState>& Dst,
                                   GRExprEngine& Engine,
                                   GRStmtNodeBuilder<GRState>& Builder,
                                   ObjCMessageExpr* ME,
                                   ExplodedNode<GRState>* Pred) {}
  
  // Stores.
  
  /// EvalStore - Evaluate the effects of a store, creating a new node
  ///  the represents the effect of binding 'Val' to the location 'TargetLV'.
  //   TargetLV is guaranteed to either be an UnknownVal or an Loc.
  virtual void EvalStore(ExplodedNodeSet<GRState>& Dst,
                         GRExprEngine& Engine,
                         GRStmtNodeBuilder<GRState>& Builder,
                         Expr* E, ExplodedNode<GRState>* Pred,
                         const GRState* St, SVal TargetLV, SVal Val);
                         
  
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
                                       SVal Cond, bool Assumption,
                                       bool& isFeasible) {
    return St;
  }
};
  
} // end clang namespace

#endif
