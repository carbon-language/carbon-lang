//== TransferFuncs.h - Path-Sens. Transfer Functions Interface ---*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines TransferFuncs, which provides a base-class that
//  defines an interface for transfer functions used by ExprEngine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GR_TRANSFERFUNCS
#define LLVM_CLANG_GR_TRANSFERFUNCS

#include "clang/GR/PathSensitive/GRState.h"
#include "clang/GR/PathSensitive/SVals.h"
#include <vector>

namespace clang {
class ObjCMessageExpr;

namespace GR {
class ExplodedNode;
class ExplodedNodeSet;
class EndPathNodeBuilder;
class ExprEngine;
class StmtNodeBuilder;
class StmtNodeBuilderRef;

class TransferFuncs {
public:
  TransferFuncs() {}
  virtual ~TransferFuncs() {}

  virtual void RegisterPrinters(std::vector<GRState::Printer*>& Printers) {}
  virtual void RegisterChecks(ExprEngine& Eng) {}


  // Calls.

  virtual void evalCall(ExplodedNodeSet& Dst,
                        ExprEngine& Engine,
                        StmtNodeBuilder& Builder,
                        const CallExpr* CE, SVal L,
                        ExplodedNode* Pred) {}

  virtual void evalObjCMessageExpr(ExplodedNodeSet& Dst,
                                   ExprEngine& Engine,
                                   StmtNodeBuilder& Builder,
                                   const ObjCMessageExpr* ME,
                                   ExplodedNode* Pred,
                                   const GRState *state) {}

  // Stores.

  virtual void evalBind(StmtNodeBuilderRef& B, SVal location, SVal val) {}

  // End-of-path and dead symbol notification.

  virtual void evalEndPath(ExprEngine& Engine,
                           EndPathNodeBuilder& Builder) {}


  virtual void evalDeadSymbols(ExplodedNodeSet& Dst,
                               ExprEngine& Engine,
                               StmtNodeBuilder& Builder,
                               ExplodedNode* Pred,
                               const GRState* state,
                               SymbolReaper& SymReaper) {}

  // Return statements.
  virtual void evalReturn(ExplodedNodeSet& Dst,
                          ExprEngine& Engine,
                          StmtNodeBuilder& Builder,
                          const ReturnStmt* S,
                          ExplodedNode* Pred) {}

  // Assumptions.
  virtual const GRState* evalAssume(const GRState *state,
                                    SVal Cond, bool Assumption) {
    return state;
  }  
};

} // end GR namespace

} // end clang namespace

#endif
