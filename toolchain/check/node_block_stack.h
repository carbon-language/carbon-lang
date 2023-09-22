// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_NODE_BLOCK_STACK_H_
#define CARBON_TOOLCHAIN_CHECK_NODE_BLOCK_STACK_H_

#include "llvm/ADT/SmallVector.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/node.h"

namespace Carbon::Check {

// A stack of node blocks that are currently being constructed in a Context. The
// contents of the node blocks are stored here until the node block is popped
// from the stack, at which point they are transferred into the SemIR::File for
// long-term storage.
//
// All pushes and pops will be vlogged.
class NodeBlockStack {
 public:
  explicit NodeBlockStack(llvm::StringLiteral name, SemIR::File& semantics_ir,
                          llvm::raw_ostream* vlog_stream)
      : name_(name), semantics_ir_(&semantics_ir), vlog_stream_(vlog_stream) {}

  // Pushes an existing node block.
  auto Push(SemIR::NodeBlockId id) -> void;

  // Pushes a new node block. It will be invalid unless PeekOrAdd is called in
  // order to support lazy allocation.
  auto Push() -> void { Push(SemIR::NodeBlockId::Invalid); }

  // Pushes a new unreachable code block.
  auto PushUnreachable() -> void { Push(SemIR::NodeBlockId::Unreachable); }

  // Returns the ID of the top node block, allocating one if necessary. If
  // `depth` is specified, returns the node at `depth` levels from the top of
  // the stack instead of the top block, where the top block is at depth 0.
  auto PeekOrAdd(int depth = 0) -> SemIR::NodeBlockId;

  // Pops the top node block. This will always return a valid node block;
  // SemIR::NodeBlockId::Empty is returned if one wasn't allocated.
  auto Pop() -> SemIR::NodeBlockId;

  // Pops the top node block, and discards it if it hasn't had an ID allocated.
  auto PopAndDiscard() -> void;

  // Adds the given node to the block at the top of the stack and returns its
  // ID.
  auto AddNode(SemIR::Node node) -> SemIR::NodeId {
    auto node_id = semantics_ir_->AddNodeInNoBlock(node);
    AddNodeId(node_id);
    return node_id;
  }

  // Adds the given node ID to the block at the top of the stack.
  auto AddNodeId(SemIR::NodeId node_id) -> void {
    CARBON_CHECK(!empty()) << "no current block";
    stack_[size_ - 1].content.push_back(node_id);
  }

  // Returns whether the current block is statically reachable.
  auto is_current_block_reachable() -> bool {
    return size_ != 0 &&
           stack_[size_ - 1].id != SemIR::NodeBlockId::Unreachable;
  }

  // Returns a view of the contents of the top node block on the stack.
  auto PeekCurrentBlockContents() -> llvm::ArrayRef<SemIR::NodeId> {
    CARBON_CHECK(!empty()) << "no current block";
    return stack_[size_ - 1].content;
  }

  // Prints the stack for a stack dump.
  auto PrintForStackDump(llvm::raw_ostream& output) const -> void;

  auto empty() const -> bool { return size() == 0; }
  auto size() const -> int { return size_; }

 private:
  struct StackEntry {
    // Preallocate an arbitrary size for the stack entries.
    // TODO: Perform measurements to pick a good starting size to avoid
    // reallocation.
    StackEntry() { content.reserve(32); }

    auto Reset(SemIR::NodeBlockId new_id) {
      id = new_id;
      content.clear();
    }

    // The block ID, if one has been allocated, Invalid if no block has been
    // allocated, or Unreachable if this block is known to be unreachable.
    SemIR::NodeBlockId id = SemIR::NodeBlockId::Invalid;

    // The content of the block. Stored as a vector rather than as a SmallVector
    // to reduce the cost of resizing `stack_` and performing swaps.
    std::vector<SemIR::NodeId> content;
  };

  // A name for debugging.
  llvm::StringLiteral name_;

  // The underlying SemIR::File instance. Always non-null.
  SemIR::File* semantics_ir_;

  // Whether to print verbose output.
  llvm::raw_ostream* vlog_stream_;

  // The actual stack.
  llvm::SmallVector<StackEntry> stack_;

  // The size of the stack. Entries after this in `stack_` are kept around so
  // that we can reuse the allocated buffer for their content.
  int size_ = 0;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_NODE_BLOCK_STACK_H_
