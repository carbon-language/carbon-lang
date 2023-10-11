// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_PENDING_BLOCK_H_
#define CARBON_TOOLCHAIN_CHECK_PENDING_BLOCK_H_

#include "llvm/ADT/SmallVector.h"
#include "toolchain/check/context.h"

namespace Carbon::Check {

// A block of code that contains pending instructions that might be needed but
// that haven't been inserted yet.
class PendingBlock {
 public:
  PendingBlock(Context& context) : context_(context) {}

  PendingBlock(const PendingBlock&) = delete;
  PendingBlock& operator=(const PendingBlock&) = delete;

  // A scope in which we will tentatively add nodes to a pending block. If we
  // leave the scope without inserting or merging the block, nodes added after
  // this point will be removed again.
  class DiscardUnusedNodesScope {
   public:
    // If `block` is not null, enters the scope. If `block` is null, this object
    // has no effect.
    DiscardUnusedNodesScope(PendingBlock* block)
        : block_(block), size_(block ? block->nodes_.size() : 0) {}
    ~DiscardUnusedNodesScope() {
      if (block_ && block_->nodes_.size() > size_) {
        block_->nodes_.truncate(size_);
      }
    }

   private:
    PendingBlock* block_;
    size_t size_;
  };

  auto AddNode(SemIR::Node node) -> SemIR::NodeId {
    auto node_id = context_.semantics_ir().AddNodeInNoBlock(node);
    nodes_.push_back(node_id);
    return node_id;
  }

  // Insert the pending block of code at the current position.
  auto InsertHere() -> void {
    for (auto id : nodes_) {
      context_.node_block_stack().AddNodeId(id);
    }
    nodes_.clear();
  }

  // Replace the node at target_id with the nodes in this block. The new value
  // for target_id should be value_id.
  auto MergeReplacing(SemIR::NodeId target_id, SemIR::NodeId value_id) -> void {
    auto value = context_.semantics_ir().GetNode(value_id);

    // There are three cases here:

    if (nodes_.empty()) {
      // 1) The block is empty. Replace `target_id` with an empty splice
      // pointing at `value_id`.
      context_.semantics_ir().ReplaceNode(
          target_id, SemIR::SpliceBlock(value.parse_node(), value.type_id(),
                                        SemIR::NodeBlockId::Empty, value_id));
    } else if (nodes_.size() == 1 && nodes_[0] == value_id) {
      // 2) The block is {value_id}. Replace `target_id` with the node referred
      // to by `value_id`. This is intended to be the common case.
      context_.semantics_ir().ReplaceNode(target_id, value);
    } else {
      // 3) Anything else: splice it into the IR, replacing `target_id`.
      context_.semantics_ir().ReplaceNode(
          target_id,
          SemIR::SpliceBlock(value.parse_node(), value.type_id(),
                             context_.semantics_ir().AddNodeBlock(nodes_),
                             value_id));
    }

    // Prepare to stash more pending instructions.
    nodes_.clear();
  }

 private:
  Context& context_;
  llvm::SmallVector<SemIR::NodeId> nodes_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_PENDING_BLOCK_H_
