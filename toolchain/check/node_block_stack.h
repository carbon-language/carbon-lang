// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_NODE_BLOCK_STACK_H_
#define CARBON_TOOLCHAIN_CHECK_NODE_BLOCK_STACK_H_

#include "llvm/ADT/SmallVector.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/node.h"

namespace Carbon::Check {

// Wraps the stack of node blocks for Context.
//
// All pushes and pops will be vlogged.
class NodeBlockStack {
 public:
  explicit NodeBlockStack(llvm::StringLiteral name, SemIR::File& semantics_ir,
                          llvm::raw_ostream* vlog_stream)
      : name_(name), semantics_ir_(&semantics_ir), vlog_stream_(vlog_stream) {}

  // Pushes an existing node block.
  auto Push(SemIR::NodeBlockId id) -> void;

  // Pushes a new node block. It will be invalid unless PeekForAdd is called in
  // order to support lazy allocation.
  auto Push() -> void { Push(SemIR::NodeBlockId::Invalid); }

  // Pushes a new unreachable code block.
  auto PushUnreachable() -> void { Push(SemIR::NodeBlockId::Unreachable); }

  // Allocates and pushes a new node block.
  auto PushForAdd() -> SemIR::NodeBlockId {
    Push();
    return PeekForAdd();
  }

  // Peeks at the top node block. This does not trigger lazy allocation, so the
  // returned node block may be invalid.
  auto Peek() -> SemIR::NodeBlockId {
    CARBON_CHECK(!empty()) << "no current block";
    return stack_[size() - 1].id;
  }

  // Returns the top node block, allocating one if it's still invalid. If
  // `depth` is specified, returns the node at `depth` levels from the top of
  // the stack, where the top block is at depth 0.
  auto PeekForAdd(int depth = 0) -> SemIR::NodeBlockId;

  // Pops the top node block. This will always return a valid node block;
  // SemIR::NodeBlockId::Empty is returned if one wasn't allocated.
  auto Pop() -> SemIR::NodeBlockId;

  // Adds the given node ID to the block at the top of the stack.
  auto AddNodeId(SemIR::NodeId node_id) -> void {
    CARBON_CHECK(!empty()) << "no current block";
    stack_[size_ - 1].content.push_back(node_id);
  }

  // Returns whether the current block is statically reachable.
  auto is_current_block_reachable() -> bool {
    return Peek() != SemIR::NodeBlockId::Unreachable;
  }

  // Prints the stack for a stack dump.
  auto PrintForStackDump(llvm::raw_ostream& output) const -> void;

  auto empty() const -> bool { return size() == 0; }
  auto size() const -> size_t { return size_; }

 private:
  struct StackEntry {
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
  size_t size_ = 0;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_NODE_BLOCK_STACK_H_
