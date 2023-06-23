// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWERING_LOWERING_BLOCK_WORKLIST_H_
#define CARBON_TOOLCHAIN_LOWERING_LOWERING_BLOCK_WORKLIST_H_

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

// A worklist for blocks that need to be lowered.
//
// The blocks form a tree, where the sequence of blocks that are pushed
// following a `Pop` that returned block B are the children of B. Blocks are
// popped in a preorder depth-first traversal over this tree, where blocks that
// are children of the same block are popped in the same order in which they
// were pushed.
//
// This traversal order is intended to produce readable IR:
//
// - In the absence of control flow back-edges, branches will typically branch
//   to blocks emitted later, although this is not guaranteed.
// - A branch and the blocks that it branches to will typically be placed close
//   together.
class LoweringBlockWorklist {
 public:
  // Add a block to the work list.
  auto Push(SemanticsNodeBlockId id) -> void { worklist_.push_back(id); }

  // Pop the next block to lower.
  auto Pop() -> SemanticsNodeBlockId {
    // Reverse the order of the blocks added since the last `Pop`, so that we
    // pop them in the order that they were `Pushed` in.
    std::reverse(worklist_.begin() + size_after_last_pop_, worklist_.end());
    SemanticsNodeBlockId result = worklist_.pop_back_val();
    size_after_last_pop_ = worklist_.size();
    return result;
  }

  auto empty() -> bool { return worklist_.empty(); }

 private:
  llvm::SmallVector<SemanticsNodeBlockId> worklist_;
  int size_after_last_pop_ = 0;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_LOWERING_LOWERING_BLOCK_WORKLIST_H_
