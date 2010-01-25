//== GRSubEngine.h - Interface of the subengine of GRCoreEngine ----*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface of a subengine of the GRCoreEngine.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_ANALYSIS_GRSUBENGINE_H
#define LLVM_CLANG_ANALYSIS_GRSUBENGINE_H

#include "clang/Checker/PathSensitive/SVals.h"

namespace clang {

class Stmt;
class CFGBlock;
class CFGElement;
class GRState;
class GRStateManager;
class GRBlockCounter;
class GRStmtNodeBuilder;
class GRBranchNodeBuilder;
class GRIndirectGotoNodeBuilder;
class GRSwitchNodeBuilder;
class GREndPathNodeBuilder;
class LocationContext;

class GRSubEngine {
public:
  virtual ~GRSubEngine() {}

  virtual const GRState* getInitialState(const LocationContext *InitLoc) = 0;

  virtual GRStateManager& getStateManager() = 0;

  /// ProcessStmt - Called by GRCoreEngine. Used to generate new successor
  ///  nodes by processing the 'effects' of a block-level statement.
  virtual void ProcessStmt(CFGElement E, GRStmtNodeBuilder& builder) = 0;

  /// ProcessBlockEntrance - Called by GRCoreEngine when start processing
  ///  a CFGBlock.  This method returns true if the analysis should continue
  ///  exploring the given path, and false otherwise.
  virtual bool ProcessBlockEntrance(CFGBlock* B, const GRState* St,
                                    GRBlockCounter BC) = 0;

  /// ProcessBranch - Called by GRCoreEngine.  Used to generate successor
  ///  nodes by processing the 'effects' of a branch condition.
  virtual void ProcessBranch(Stmt* Condition, Stmt* Term,
                             GRBranchNodeBuilder& builder) = 0;

  /// ProcessIndirectGoto - Called by GRCoreEngine.  Used to generate successor
  ///  nodes by processing the 'effects' of a computed goto jump.
  virtual void ProcessIndirectGoto(GRIndirectGotoNodeBuilder& builder) = 0;

  /// ProcessSwitch - Called by GRCoreEngine.  Used to generate successor
  ///  nodes by processing the 'effects' of a switch statement.
  virtual void ProcessSwitch(GRSwitchNodeBuilder& builder) = 0;

  /// ProcessEndPath - Called by GRCoreEngine.  Used to generate end-of-path
  ///  nodes when the control reaches the end of a function.
  virtual void ProcessEndPath(GREndPathNodeBuilder& builder) = 0;
  
  /// EvalAssume - Called by ConstraintManager. Used to call checker-specific
  ///  logic for handling assumptions on symbolic values.
  virtual const GRState* ProcessAssume(const GRState *state,
                                       SVal cond, bool assumption) = 0;
};
}

#endif
