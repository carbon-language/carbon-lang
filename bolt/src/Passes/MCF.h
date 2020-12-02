//===--- Passes/MCF.h -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_MCF_H
#define LLVM_TOOLS_LLVM_BOLT_MCF_H

namespace llvm {
namespace bolt {

class BinaryFunction;

enum MCFCostFunction : char {
  MCF_DISABLE = 0,
  MCF_LINEAR,
  MCF_QUADRATIC,
  MCF_LOG,
  MCF_BLAMEFTS
};

/// Fill edge counts based on the basic block count. Used in nonLBR mode when
/// we only have bb count.
void estimateEdgeCounts(BinaryFunction &BF);

/// Entry point for computing a min-cost flow for the CFG with the goal
/// of fixing the flow of the CFG edges, that is, making sure it obeys the
/// flow-conservation equation  SumInEdges = SumOutEdges.
///
/// To do this, we create an instance of the min-cost flow problem in a
/// similar way as the one discussed in the work of Roy Levin "Completing
/// Incomplete Edge Profile by Applying Minimum Cost Circulation Algorithms".
/// We do a few things differently, though. We don't populate edge counts using
/// weights coming from a static branch prediction technique and we don't
/// use the same cost function.
///
/// If cost function BlameFTs is used, assign all remaining flow to
/// fall-throughs. This is used when the sampling is based on taken branches
/// that do not account for them.
void solveMCF(BinaryFunction &BF, MCFCostFunction CostFunction);

}
}


#endif
