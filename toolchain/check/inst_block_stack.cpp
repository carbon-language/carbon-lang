// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/inst_block_stack.h"

#include "common/vlog.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

namespace Carbon::Check {

auto InstBlockStack::Push(SemIR::InstBlockId id) -> void {
  CARBON_VLOG() << name_ << " Push " << id_stack_.size() << "\n";
  CARBON_CHECK(id_stack_.size() < (1 << 20))
      << "Excessive stack size: likely infinite loop";
  id_stack_.push_back(id);
  insts_stack_.PushArray();
}

auto InstBlockStack::Push(SemIR::InstBlockId id,
                          llvm::ArrayRef<SemIR::InstId> inst_ids) -> void {
  Push(id);
  insts_stack_.AppendToTop(inst_ids);
}

auto InstBlockStack::PeekOrAdd(int depth) -> SemIR::InstBlockId {
  CARBON_CHECK(static_cast<int>(id_stack_.size()) > depth) << "no such block";
  int index = id_stack_.size() - depth - 1;
  auto& slot = id_stack_[index];
  if (!slot.is_valid()) {
    slot = sem_ir_->inst_blocks().AddDefaultValue();
  }
  return slot;
}

auto InstBlockStack::Pop() -> SemIR::InstBlockId {
  CARBON_CHECK(!empty()) << "no current block";
  auto id = id_stack_.pop_back_val();
  auto insts = insts_stack_.PeekArray();

  // Finalize the block.
  if (!insts.empty() && id != SemIR::InstBlockId::Unreachable) {
    if (id.is_valid()) {
      sem_ir_->inst_blocks().Set(id, insts);
    } else {
      id = sem_ir_->inst_blocks().Add(insts);
    }
  }

  insts_stack_.PopArray();

  CARBON_VLOG() << name_ << " Pop " << id_stack_.size() << ": " << id << "\n";
  return id.is_valid() ? id : SemIR::InstBlockId::Empty;
}

auto InstBlockStack::PopAndDiscard() -> void {
  CARBON_CHECK(!empty()) << "no current block";
  id_stack_.pop_back();
  insts_stack_.PopArray();
  CARBON_VLOG() << name_ << " PopAndDiscard " << id_stack_.size() << "\n";
}

auto InstBlockStack::PrintForStackDump(llvm::raw_ostream& output) const
    -> void {
  output << name_ << ":\n";
  for (const auto& [i, id] : llvm::enumerate(id_stack_)) {
    output << "\t" << i << ".\t" << id << "\t{";
    llvm::ListSeparator sep;
    for (auto id : insts_stack_.PeekArrayAt(i)) {
      output << sep << id;
    }
    output << "}\n";
  }
}

}  // namespace Carbon::Check
