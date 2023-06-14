// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWERING_LOWERING_BLOCK_WORKLIST_H_
#define CARBON_TOOLCHAIN_LOWERING_LOWERING_BLOCK_WORKLIST_H_

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

// A worklist for blocks that need to be lowered. Blocks are popped from the
// worklist in the order in which they are first pushed, in a depth-first
// traversal.
class LoweringBlockWorklist {
 public:
  // Add a block to the work list, if it's not already present.
  auto Push(SemanticsNodeBlockId id) -> void {
    if (found_.insert(id).second) {
      worklist_.push_back(id);
    }
  }

  // Pop the next block to lower.
  auto Pop() -> SemanticsNodeBlockId {
    std::reverse(worklist_.begin() + size_after_last_pop_, worklist_.end());
    SemanticsNodeBlockId result = worklist_.pop_back_val();
    size_after_last_pop_ = worklist_.size();
    return result;
  }

  auto Empty() -> bool { return worklist_.empty(); }

 private:
  llvm::SmallVector<SemanticsNodeBlockId> worklist_;
  llvm::DenseSet<SemanticsNodeBlockId> found_;
  int size_after_last_pop_ = 0;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_LOWERING_LOWERING_BLOCK_WORKLIST_H_
