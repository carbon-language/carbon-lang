//== SubEngine.h - Interface of the subengine of CoreEngine --------*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface of a subengine of the CoreEngine.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_GR_SUBENGINE_H
#define LLVM_CLANG_GR_SUBENGINE_H

#include "clang/Analysis/ProgramPoint.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/Store.h"

namespace clang {

class CFGBlock;
class CFGElement;
class LocationContext;
class Stmt;

namespace ento {
  
template <typename PP> class GenericNodeBuilder;
class AnalysisManager;
class ExplodedNodeSet;
class ExplodedNode;
class GRState;
class GRStateManager;
class BlockCounter;
class StmtNodeBuilder;
class BranchNodeBuilder;
class IndirectGotoNodeBuilder;
class SwitchNodeBuilder;
class EndOfFunctionNodeBuilder;
class CallEnterNodeBuilder;
class CallExitNodeBuilder;
class MemRegion;

class SubEngine {
public:
  virtual ~SubEngine() {}

  virtual const GRState* getInitialState(const LocationContext *InitLoc) = 0;

  virtual AnalysisManager &getAnalysisManager() = 0;

  virtual GRStateManager &getStateManager() = 0;

  /// Called by CoreEngine. Used to generate new successor
  /// nodes by processing the 'effects' of a block-level statement.
  virtual void processCFGElement(const CFGElement E, StmtNodeBuilder& builder)=0;

  /// Called by CoreEngine when it starts processing a CFGBlock.  The
  /// SubEngine is expected to populate dstNodes with new nodes representing
  /// updated analysis state, or generate no nodes at all if it doesn't.
  virtual void processCFGBlockEntrance(ExplodedNodeSet &dstNodes,
                            GenericNodeBuilder<BlockEntrance> &nodeBuilder) = 0;

  /// Called by CoreEngine.  Used to generate successor
  ///  nodes by processing the 'effects' of a branch condition.
  virtual void processBranch(const Stmt* Condition, const Stmt* Term,
                             BranchNodeBuilder& builder) = 0;

  /// Called by CoreEngine.  Used to generate successor
  /// nodes by processing the 'effects' of a computed goto jump.
  virtual void processIndirectGoto(IndirectGotoNodeBuilder& builder) = 0;

  /// Called by CoreEngine.  Used to generate successor
  /// nodes by processing the 'effects' of a switch statement.
  virtual void processSwitch(SwitchNodeBuilder& builder) = 0;

  /// Called by CoreEngine.  Used to generate end-of-path
  /// nodes when the control reaches the end of a function.
  virtual void processEndOfFunction(EndOfFunctionNodeBuilder& builder) = 0;

  // Generate the entry node of the callee.
  virtual void processCallEnter(CallEnterNodeBuilder &builder) = 0;

  // Generate the first post callsite node.
  virtual void processCallExit(CallExitNodeBuilder &builder) = 0;

  /// Called by ConstraintManager. Used to call checker-specific
  /// logic for handling assumptions on symbolic values.
  virtual const GRState* processAssume(const GRState *state,
                                       SVal cond, bool assumption) = 0;

  /// wantsRegionChangeUpdate - Called by GRStateManager to determine if a
  ///  region change should trigger a processRegionChanges update.
  virtual bool wantsRegionChangeUpdate(const GRState* state) = 0;

  /// processRegionChanges - Called by GRStateManager whenever a change is made
  ///  to the store. Used to update checkers that track region values.
  virtual const GRState *
  processRegionChanges(const GRState *state,
                       const StoreManager::InvalidatedSymbols *invalidated,
                       const MemRegion* const *Begin,
                       const MemRegion* const *End) = 0;


  inline const GRState *
  processRegionChange(const GRState* state,
                      const MemRegion* MR) {
    return processRegionChanges(state, 0, &MR, &MR+1);
  }

  /// Called by CoreEngine when the analysis worklist is either empty or the
  //  maximum number of analysis steps have been reached.
  virtual void processEndWorklist(bool hasWorkRemaining) = 0;
};

} // end GR namespace

} // end clang namespace

#endif
