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

  ~SemanticsNodeBlockStack() { CARBON_CHECK(stack_.empty()) << stack_.size(); }

  auto Push() -> void;

  // TODO: Try to remove this in favor of the lazy alloc in Push.
  auto PushWithUnconditionalAlloc() -> SemanticsNodeBlockId;

  auto Peek() -> SemanticsNodeBlockId { return stack_.back(); }
  auto PeekForAdd() -> SemanticsNodeBlockId;

  auto Pop() -> SemanticsNodeBlockId;

  // Prints the stack for a stack dump.
  auto PrintForStackDump(llvm::raw_ostream& output) const -> void;

 private:
  llvm::SmallVector<llvm::SmallVector<SemanticsNodeId>>* node_blocks_;

  // Whether to print verbose output.
  llvm::raw_ostream* vlog_stream_;

  // The actual stack.
  // PushEntry and PopEntry control modification in order to centralize
  // vlogging.
  llvm::SmallVector<SemanticsNodeBlockId> stack_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_BLOCK_STACK_H_
