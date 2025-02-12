// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_REGION_STACK_H_
#define CARBON_TOOLCHAIN_CHECK_REGION_STACK_H_

#include <utility>

#include "common/array_stack.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Provides a stack of single-entry regions being built.
class RegionStack {
 public:
  // A callback for Context::TODO.
  using TodoFn = std::function<auto(SemIRLoc, std::string)->void>;

  explicit RegionStack(TodoFn todo_fn) : todo_fn_(std::move(todo_fn)) {}

  // Mark the start of a new single-entry region with the given entry block.
  auto PushRegion(SemIR::InstBlockId entry_block_id) -> void {
    stack_.PushArray();
    stack_.AppendToTop(entry_block_id);
  }

  // Add `block_id` to the most recently pushed single-entry region. To preserve
  // the single-entry property, `block_id` must not be directly reachable from
  // any block outside the region. To ensure the region's blocks are in lexical
  // order, this should be called when the first parse node associated with this
  // block is handled, or as close as possible.
  auto AddToRegion(SemIR::InstBlockId block_id, SemIR::LocId loc_id) -> void {
    if (stack_.empty()) {
      todo_fn_(loc_id,
               "Control flow expressions are currently only supported inside "
               "functions.");
      return;
    }
    if (block_id == SemIR::InstBlockId::Unreachable) {
      return;
    }

    stack_.AppendToTop(block_id);
  }

  // Adds multiple blocks at once. The caller is responsible for validating that
  // each block is reachable.
  auto AddToRegion(llvm::ArrayRef<SemIR::InstBlockId> block_ids) -> void {
    stack_.AppendToTop(block_ids);
  }

  // Complete creation of the most recently pushed single-entry region, and
  // return a list of its blocks.
  auto PopRegion() -> llvm::SmallVector<SemIR::InstBlockId> {
    llvm::SmallVector<SemIR::InstBlockId> result(stack_.PeekArray());
    stack_.PopArray();
    return result;
  }

  // Pops a region, and does not return its contents.
  auto PopAndDiscardRegion() -> void { stack_.PopArray(); }

  // Returns the top-most region.
  auto PeekRegion() -> llvm::ArrayRef<SemIR::InstBlockId> {
    return stack_.PeekArray();
  }

  // Returns true if any regions have been added.
  auto empty() -> bool { return stack_.empty(); }

 private:
  TodoFn todo_fn_;

  ArrayStack<SemIR::InstBlockId> stack_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_REGION_STACK_H_
