//===--- Passes/CallGraphWalker.h -----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_CALLGRAPHWALKER_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_CALLGRAPHWALKER_H

#include "BinaryContext.h"
#include "BinaryFunction.h"
#include "BinaryFunctionCallGraph.h"
#include <queue>
#include <map>
#include <set>
#include <vector>

namespace llvm {
namespace bolt {

/// Perform a bottom-up walk of the call graph with the intent of computing
/// a property that depends on callees. In the event of a CG cycles, this will
/// re-visit functions until their observed property converges.
class CallGraphWalker {
  BinaryFunctionCallGraph &CG;

  /// DFS or reverse post-ordering of the call graph nodes to allow us to
  /// traverse the call graph bottom-up
  std::deque<BinaryFunction *> TopologicalCGOrder;

  /// Stores all visitor functions to call when traversing the call graph
  typedef std::function<bool(BinaryFunction*)> CallbackTy;
  std::vector<CallbackTy> Visitors;

  /// Do the bottom-up traversal
  void traverseCG();

public:
  /// Initialize core context references but don't do anything yet
  CallGraphWalker(BinaryFunctionCallGraph &CG) : CG(CG) {}

  /// Register a new callback function to be called for each function when
  /// traversing the call graph bottom-up. Function should return true iff
  /// whatever information it is keeping track of has changed. Function must
  /// converge with time, ie, it must eventually return false, otherwise the
  /// call graph walk will never finish.
  void registerVisitor(CallbackTy Callback) {
    Visitors.emplace_back(Callback);
  }

  /// Build the call graph, establish a traversal order and traverse it.
  void walk();
};

}
}

#endif
