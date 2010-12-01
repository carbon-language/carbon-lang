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

#include "clang/Checker/PathSensitive/GRState.h"
#include "clang/Checker/PathSensitive/SVals.h"
#include <vector>

namespace clang {
class ExplodedNode;
class ExplodedNodeSet;
class GREndPathNodeBuilder;
class GRExprEngine;
class GRStmtNodeBuilder;
class GRStmtNodeBuilderRef;
class ObjCMessageExpr;

class GRTransferFuncs {
public:
  GRTransferFuncs() {}
  virtual ~GRTransferFuncs() {}

  virtual void RegisterPrinters(std::vector<GRState::Printer*>& Printers) {}
  virtual void RegisterChecks(GRExprEngine& Eng) {}


  // Calls.

  virtual void evalCall(ExplodedNodeSet& Dst,
                        GRExprEngine& Engine,
                        GRStmtNodeBuilder& Builder,
                        const CallExpr* CE, SVal L,
                        ExplodedNode* Pred) {}

  virtual void evalObjCMessageExpr(ExplodedNodeSet& Dst,
                                   GRExprEngine& Engine,
                                   GRStmtNodeBuilder& Builder,
                                   const ObjCMessageExpr* ME,
                                   ExplodedNode* Pred,
                                   const GRState *state) {}

  // Stores.

  virtual void evalBind(GRStmtNodeBuilderRef& B, SVal location, SVal val) {}

  // End-of-path and dead symbol notification.

  virtual void evalEndPath(GRExprEngine& Engine,
                           GREndPathNodeBuilder& Builder) {}


  virtual void evalDeadSymbols(ExplodedNodeSet& Dst,
                               GRExprEngine& Engine,
                               GRStmtNodeBuilder& Builder,
                               ExplodedNode* Pred,
                               const GRState* state,
                               SymbolReaper& SymReaper) {}

  // Return statements.
  virtual void evalReturn(ExplodedNodeSet& Dst,
                          GRExprEngine& Engine,
                          GRStmtNodeBuilder& Builder,
                          const ReturnStmt* S,
                          ExplodedNode* Pred) {}

  // Assumptions.
  virtual const GRState* evalAssume(const GRState *state,
                                    SVal Cond, bool Assumption) {
    return state;
  }  
};
} // end clang namespace

#endif
