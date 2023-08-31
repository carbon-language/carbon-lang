// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/node_block_stack.h"

#include "common/vlog.h"
#include "llvm/ADT/STLExtras.h"
#include "toolchain/sem_ir/node.h"

namespace Carbon::Check {

auto NodeBlockStack::Push(SemIR::NodeBlockId id) -> void {
  CARBON_VLOG() << name_ << " Push " << stack_.size() << "\n";
  CARBON_CHECK(stack_.size() < (1 << 20))
      << "Excessive stack size: likely infinite loop";
  stack_.push_back(id);
}

auto NodeBlockStack::PeekForAdd() -> SemIR::NodeBlockId {
  CARBON_CHECK(!stack_.empty()) << "no current block";
  auto& back = stack_.back();
  if (!back.is_valid()) {
    back = semantics_ir_->AddNodeBlock();
    CARBON_VLOG() << name_ << " Add " << stack_.size() - 1 << ": " << back
                  << "\n";
  }
  return back;
}

auto NodeBlockStack::Pop() -> SemIR::NodeBlockId {
  CARBON_CHECK(!stack_.empty()) << "no current block";
  auto back = stack_.pop_back_val();
  CARBON_VLOG() << name_ << " Pop " << stack_.size() << ": " << back << "\n";
  if (!back.is_valid()) {
    return SemIR::NodeBlockId::Empty;
  }
  return back;
}

auto NodeBlockStack::PrintForStackDump(llvm::raw_ostream& output) const
    -> void {
  output << name_ << ":\n";
  for (auto [i, entry] : llvm::enumerate(stack_)) {
    output << "\t" << i << ".\t" << entry << "\n";
  }
}

}  // namespace Carbon::Check
