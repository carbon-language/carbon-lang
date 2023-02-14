// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_node_block_stack.h"

#include "common/vlog.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

auto SemanticsNodeBlockStack::Push() -> void {
  CARBON_VLOG() << name_ << " Push " << stack_.size() << "\n";
  CARBON_CHECK(stack_.size() < (1 << 20))
      << "Excessive stack size: likely infinite loop";
  stack_.push_back(SemanticsNodeBlockId::Invalid);
}

auto SemanticsNodeBlockStack::PeekForAdd() -> SemanticsNodeBlockId {
  CARBON_CHECK(!stack_.empty());
  auto& back = stack_.back();
  if (!back.is_valid()) {
    SemanticsNodeBlockId block_id(node_blocks_->size());
    node_blocks_->resize(block_id.index + 1);
    back = block_id;
    CARBON_VLOG() << name_ << " Add " << stack_.size() - 1 << ": " << back
                  << "\n";
  }
  return back;
}

auto SemanticsNodeBlockStack::Pop() -> SemanticsNodeBlockId {
  auto back = stack_.pop_back_val();
  CARBON_VLOG() << name_ << " Pop " << stack_.size() << ": " << back << "\n";
  if (!back.is_valid()) {
    return SemanticsNodeBlockId::Empty;
  }
  return back;
}

auto SemanticsNodeBlockStack::PrintForStackDump(llvm::raw_ostream& output) const
    -> void {
  output << name_ << ":\n";
  for (int i = 0; i < static_cast<int>(stack_.size()); ++i) {
    output << "\t" << i << ".\t" << stack_[i] << "\n";
  }
}

}  // namespace Carbon
