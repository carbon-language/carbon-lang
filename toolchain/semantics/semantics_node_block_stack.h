// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_BLOCK_STACK_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_BLOCK_STACK_H_

#include <type_traits>

#include "common/check.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

// Wraps the stack of node blocks for SemanticsParseTreeHandler.
//
// All pushes and pops will be vlogged.
class SemanticsNodeBlockStack {
 public:
  explicit SemanticsNodeBlockStack(
      llvm::SmallVector<llvm::SmallVector<SemanticsNodeId>>& node_blocks,
      llvm::raw_ostream* vlog_stream)
      : node_blocks_(&node_blocks), vlog_stream_(vlog_stream) {}

  // Pushes a new node block. It will be invalid unless PeekForAdd is called in
  // order to support lazy allocation.
  auto Push() -> void;

  // Peeks at the top node block. This does not trigger lazy allocation, so the
  // returned node block may be invalid.
  auto Peek() -> SemanticsNodeBlockId { return stack_.back(); }

  // Returns the top node block, allocating one if it's still invalid.
  auto PeekForAdd() -> SemanticsNodeBlockId;

  // Pops the top node block. This will always return a valid node block;
  // SemanticsNodeBlockId::Empty is returned if one wasn't allocated.
  auto Pop() -> SemanticsNodeBlockId;

  // Prints the stack for a stack dump.
  auto PrintForStackDump(llvm::raw_ostream& output) const -> void;

  auto empty() const -> bool { return stack_.empty(); }
  auto size() const -> size_t { return stack_.size(); }

 private:
  // The underlying node block storage on SemanticsIR. Always non-null.
  llvm::SmallVector<llvm::SmallVector<SemanticsNodeId>>* const node_blocks_;

  // Whether to print verbose output.
  llvm::raw_ostream* vlog_stream_;

  // The actual stack.
  // PushEntry and PopEntry control modification in order to centralize
  // vlogging.
  llvm::SmallVector<SemanticsNodeBlockId> stack_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_BLOCK_STACK_H_
