// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_PENDING_BLOCK_H_
#define CARBON_TOOLCHAIN_CHECK_PENDING_BLOCK_H_

#include "llvm/ADT/SmallVector.h"
#include "toolchain/check/context.h"
#include "toolchain/check/inst.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

// A block of code that contains pending instructions that might be needed but
// that haven't been inserted yet.
class PendingBlock {
 public:
  explicit PendingBlock(Context& context) : context_(context) {}

  PendingBlock(const PendingBlock&) = delete;
  auto operator=(const PendingBlock&) -> PendingBlock& = delete;

  // A scope in which we will tentatively add instructions to a pending block.
  // If we leave the scope without inserting or merging the block, instructions
  // added after this point will be removed again.
  class DiscardUnusedInstsScope {
   public:
    // If `block` is not null, enters the scope. If `block` is null, this object
    // has no effect.
    explicit DiscardUnusedInstsScope(PendingBlock* block)
        : block_(block), size_(block ? block->insts_.size() : 0) {}
    ~DiscardUnusedInstsScope() {
      if (block_ && block_->insts_.size() > size_) {
        block_->insts_.truncate(size_);
      }
    }

   private:
    PendingBlock* block_;
    size_t size_;
  };

  template <typename InstT, typename LocT>
  auto AddInst(LocT loc_id, InstT inst) -> SemIR::InstId {
    auto inst_id = AddInstInNoBlock(context_, loc_id, inst);
    insts_.push_back(inst_id);
    return inst_id;
  }

  template <typename InstT, typename LocT>
  auto AddInstWithCleanup(LocT loc_id, InstT inst) -> SemIR::InstId {
    auto inst_id = AddInstWithCleanupInNoBlock(context_, loc_id, inst);
    insts_.push_back(inst_id);
    return inst_id;
  }

  // Insert the pending block of code at the current position.
  auto InsertHere() -> void {
    for (auto id : insts_) {
      context_.inst_block_stack().AddInstId(id);
    }
    insts_.clear();
  }

  // Replace the instruction at target_id with the instructions in this block.
  // The new value for target_id should be value_id.
  auto MergeReplacing(SemIR::InstId target_id, SemIR::InstId value_id) -> void {
    SemIR::LocIdAndInst value = context_.insts().GetWithLocId(value_id);

    if (insts_.size() == 1 && insts_[0] == value_id) {
      // The block is {value_id}. Replace `target_id` with the instruction
      // referred to by `value_id`. This is intended to be the common case.
    } else {
      // Anything else: splice it into the IR, replacing `target_id`. This
      // includes empty blocks, which `Add` handles.
      value.inst =
          SemIR::SpliceBlock{.type_id = value.inst.type_id(),
                             .block_id = context_.inst_blocks().Add(insts_),
                             .result_id = value_id};
    }

    ReplaceLocIdAndInstBeforeConstantUse(context_, target_id, value);

    // Prepare to stash more pending instructions.
    insts_.clear();
  }

 private:
  Context& context_;
  llvm::SmallVector<SemIR::InstId> insts_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_PENDING_BLOCK_H_
