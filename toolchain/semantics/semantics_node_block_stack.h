// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_BLOCK_STACK_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_BLOCK_STACK_H_

#include <type_traits>

#include "llvm/ADT/SmallVector.h"
#include "toolchain/semantics/semantics_ir.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

// Wraps the stack of node blocks for SemanticsParseTreeHandler.
//
// All pushes and pops will be vlogged.
class SemanticsNodeBlockStack {
 public:
  explicit SemanticsNodeBlockStack(llvm::StringLiteral name,
                                   SemanticsIR& semantics_ir,
                                   llvm::raw_ostream* vlog_stream)
      : name_(name), semantics_ir_(&semantics_ir), vlog_stream_(vlog_stream) {}

  // Pushes an existing node block.
  auto Push(SemanticsNodeBlockId id) -> void;

  // Pushes a new node block. It will be invalid unless PeekForAdd is called in
  // order to support lazy allocation.
  auto Push() -> void { Push(SemanticsNodeBlockId::Invalid); }

  // Pushes a new unreachable code block.
  auto PushUnreachable() -> void { Push(SemanticsNodeBlockId::Unreachable); }

  // Allocates and pushes a new node block.
  auto PushForAdd() -> SemanticsNodeBlockId {
    Push();
    return PeekForAdd();
  }

  // Peeks at the top node block. This does not trigger lazy allocation, so the
  // returned node block may be invalid.
  auto Peek() -> SemanticsNodeBlockId {
    CARBON_CHECK(!stack_.empty()) << "no current block";
    return stack_.back();
  }

  // Returns the top node block, allocating one if it's still invalid.
  auto PeekForAdd() -> SemanticsNodeBlockId;

  // Pops the top node block. This will always return a valid node block;
  // SemanticsNodeBlockId::Empty is returned if one wasn't allocated.
  auto Pop() -> SemanticsNodeBlockId;

  // Pops the top node block, ensuring that it is lazily allocated if it's
  // empty. For use when more nodes will be added to the block later.
  auto PopForAdd() -> SemanticsNodeBlockId {
    PeekForAdd();
    return Pop();
  }

  // Returns whether the current block is statically reachable.
  auto is_current_block_reachable() -> bool {
    return Peek() != SemanticsNodeBlockId::Unreachable;
  }

  // Prints the stack for a stack dump.
  auto PrintForStackDump(llvm::raw_ostream& output) const -> void;

  auto empty() const -> bool { return stack_.empty(); }
  auto size() const -> size_t { return stack_.size(); }

 private:
  // A name for debugging.
  llvm::StringLiteral name_;

  // The underlying SemanticsIR instance. Always non-null.
  SemanticsIR* semantics_ir_;

  // Whether to print verbose output.
  llvm::raw_ostream* vlog_stream_;

  // The actual stack.
  // PushEntry and PopEntry control modification in order to centralize
  // vlogging.
  llvm::SmallVector<SemanticsNodeBlockId> stack_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_BLOCK_STACK_H_
