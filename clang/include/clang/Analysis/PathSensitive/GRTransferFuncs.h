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
class ObjCMessageExpr;
class GRStmtNodeBuilderRef;

class GRTransferFuncs {
public:
  GRTransferFuncs() {}
  virtual ~GRTransferFuncs() {}

  virtual void RegisterPrinters(std::vector<GRState::Printer*>& Printers) {}
  virtual void RegisterChecks(GRExprEngine& Eng) {}


  // Calls.

  virtual void EvalCall(ExplodedNodeSet& Dst,
                        GRExprEngine& Engine,
                        GRStmtNodeBuilder& Builder,
                        CallExpr* CE, SVal L,
                        ExplodedNode* Pred) {}

  virtual void EvalObjCMessageExpr(ExplodedNodeSet& Dst,
                                   GRExprEngine& Engine,
                                   GRStmtNodeBuilder& Builder,
                                   ObjCMessageExpr* ME,
                                   ExplodedNode* Pred,
                                   const GRState *state) {}

  // Stores.

  virtual void EvalBind(GRStmtNodeBuilderRef& B, SVal location, SVal val) {}

  // End-of-path and dead symbol notification.

  virtual void EvalEndPath(GRExprEngine& Engine,
                           GREndPathNodeBuilder& Builder) {}


  virtual void EvalDeadSymbols(ExplodedNodeSet& Dst,
                               GRExprEngine& Engine,
                               GRStmtNodeBuilder& Builder,
                               ExplodedNode* Pred,
                               Stmt* S, const GRState* state,
                               SymbolReaper& SymReaper) {}

  // Return statements.
  virtual void EvalReturn(ExplodedNodeSet& Dst,
                          GRExprEngine& Engine,
                          GRStmtNodeBuilder& Builder,
                          ReturnStmt* S,
                          ExplodedNode* Pred) {}

  // Assumptions.
  virtual const GRState* EvalAssume(const GRState *state,
                                    SVal Cond, bool Assumption) {
    return state;
  }  
};
} // end clang namespace

#endif
