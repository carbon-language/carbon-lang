// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/inst_block_stack.h"

#include "common/vlog.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

namespace Carbon::Check {

auto InstBlockStack::Push(SemIR::InstBlockId id) -> void {
  CARBON_VLOG() << name_ << " Push " << size_ << "\n";
  CARBON_CHECK(size_ < (1 << 20))
      << "Excessive stack size: likely infinite loop";
  if (size_ == static_cast<int>(stack_.size())) {
    stack_.emplace_back();
  }
  stack_[size_].Reset(id);
  ++size_;
}

auto InstBlockStack::PushGlobalInit() -> void {
  Push(SemIR::InstBlockId::GlobalInit);
  stack_[size_ - 1].content = std::move(init_block_);
}

auto InstBlockStack::PeekOrAdd(int depth) -> SemIR::InstBlockId {
  CARBON_CHECK(size_ > depth) << "no such block";
  int index = size_ - depth - 1;
  auto& slot = stack_[index];
  if (!slot.id.is_valid()) {
    slot.id = sem_ir_->inst_blocks().AddDefaultValue();
  }
  return slot.id;
}

auto InstBlockStack::Pop() -> SemIR::InstBlockId {
  CARBON_CHECK(!empty()) << "no current block";
  --size_;
  auto& back = stack_[size_];

  // Finalize the block.
  if (!back.content.empty() && back.id != SemIR::InstBlockId::Unreachable) {
    if (back.id.is_valid()) {
      sem_ir_->inst_blocks().Set(back.id, back.content);
    } else {
      back.id = sem_ir_->inst_blocks().Add(back.content);
    }
  }

  CARBON_VLOG() << name_ << " Pop " << size_ << ": " << back.id << "\n";
  if (!back.id.is_valid()) {
    return SemIR::InstBlockId::Empty;
  }
  return back.id;
}

auto InstBlockStack::PopGlobalInit() -> void {
  CARBON_CHECK(stack_[size_ - 1].id == SemIR::InstBlockId::GlobalInit)
      << "Trying to pop Inits block from " << name_
      << " but a different block is present!";
  init_block_ = std::move(stack_[size_ - 1].content);
  PopAndDiscard();
}

auto InstBlockStack::PopAndDiscard() -> void {
  CARBON_CHECK(!empty()) << "no current block";
  --size_;
  CARBON_VLOG() << name_ << " PopAndDiscard " << size_ << "\n";
}

auto InstBlockStack::PrintForStackDump(llvm::raw_ostream& output) const
    -> void {
  output << name_ << ":\n";
  for (const auto& [i, entry] : llvm::enumerate(stack_)) {
    output << "\t" << i << ".\t" << entry.id << "\t{";
    llvm::ListSeparator sep;
    for (auto id : entry.content) {
      output << sep << id;
    }
    output << "}\n";
  }
}

}  // namespace Carbon::Check
