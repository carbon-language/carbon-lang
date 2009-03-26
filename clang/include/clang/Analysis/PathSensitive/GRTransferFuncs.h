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

#include "clang/Analysis/PathSensitive/SVals.h"
#include "clang/Analysis/PathSensitive/GRCoreEngine.h"
#include "clang/Analysis/PathSensitive/GRState.h"
#include <vector>

namespace clang {
  
  class GRExprEngine;
  class BugReporter;
  class ObjCMessageExpr;
  class GRStmtNodeBuilderRef;
  
class GRTransferFuncs {
  friend class GRExprEngine;  
protected:
  virtual SVal DetermEvalBinOpNN(GRExprEngine& Eng,
                                 BinaryOperator::Opcode Op,
                                 NonLoc L, NonLoc R, QualType T) {
    return UnknownVal();
  }
  
public:
  GRTransferFuncs() {}
  virtual ~GRTransferFuncs() {}
  
  virtual void RegisterPrinters(std::vector<GRState::Printer*>& Printers) {}
  virtual void RegisterChecks(BugReporter& BR) {}
  
  // Casts.
  
  virtual SVal EvalCast(GRExprEngine& Engine, NonLoc V, QualType CastT) =0;  
  virtual SVal EvalCast(GRExprEngine& Engine, Loc V, QualType CastT) = 0;

  // Unary Operators.
  
  virtual SVal EvalMinus(GRExprEngine& Engine, UnaryOperator* U, NonLoc X) = 0;

  virtual SVal EvalComplement(GRExprEngine& Engine, NonLoc X) = 0;

  // Binary Operators.
  // FIXME: We're moving back towards using GREXprEngine directly.  No need
  // for OStates
  virtual void EvalBinOpNN(GRStateSet& OStates, GRExprEngine& Eng,
                           const GRState* St, Expr* Ex,
                           BinaryOperator::Opcode Op, NonLoc L, NonLoc R,
                           QualType T);
  
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
  
  virtual void EvalBind(GRStmtNodeBuilderRef& B, SVal location, SVal val) {}
  
  // End-of-path and dead symbol notification.
  
  virtual void EvalEndPath(GRExprEngine& Engine,
                           GREndPathNodeBuilder<GRState>& Builder) {}
  
  
  virtual void EvalDeadSymbols(ExplodedNodeSet<GRState>& Dst,
                               GRExprEngine& Engine,
                               GRStmtNodeBuilder<GRState>& Builder,
                               ExplodedNode<GRState>* Pred,
                               Stmt* S, const GRState* state,
                               SymbolReaper& SymReaper) {}
  
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
