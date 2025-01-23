// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_INST_BLOCK_STACK_H_
#define CARBON_TOOLCHAIN_CHECK_INST_BLOCK_STACK_H_

#include "common/array_stack.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/formatter.h"

namespace Carbon::Check {

// A stack of instruction blocks that are currently being constructed in a
// Context. The contents of the instruction blocks are stored here until the
// instruction block is popped from the stack, at which point they are
// transferred into the SemIR::File for long-term storage.
//
// All pushes and pops will be vlogged.
class InstBlockStack {
 public:
  explicit InstBlockStack(llvm::StringLiteral name, SemIR::File& sem_ir,
                          llvm::raw_ostream* vlog_stream)
      : name_(name), sem_ir_(&sem_ir), vlog_stream_(vlog_stream) {}

  // Pushes an existing instruction block.
  auto Push(SemIR::InstBlockId id) -> void;

  // Pushes an existing instruction block with a set of instructions.
  auto Push(SemIR::InstBlockId id, llvm::ArrayRef<SemIR::InstId> inst_ids)
      -> void;

  // Pushes a new instruction block. It will be `None` unless PeekOrAdd is
  // called in order to support lazy allocation.
  auto Push() -> void { Push(SemIR::InstBlockId::None); }

  // Pushes a new unreachable code block.
  auto PushUnreachable() -> void { Push(SemIR::InstBlockId::Unreachable); }

  // Returns the ID of the top instruction block, allocating one if necessary.
  // If `depth` is specified, returns the instruction at `depth` levels from the
  // top of the stack instead of the top block, where the top block is at depth
  // 0.
  auto PeekOrAdd(int depth = 0) -> SemIR::InstBlockId;

  // Pops the top instruction block. This will never return `None`; `Empty` is
  // returned if one wasn't allocated.
  auto Pop() -> SemIR::InstBlockId;

  // Pops the top instruction block, and discards it if it hasn't had an ID
  // allocated.
  auto PopAndDiscard() -> void;

  // Adds the given instruction ID to the block at the top of the stack.
  auto AddInstId(SemIR::InstId inst_id) -> void {
    CARBON_CHECK(!empty(), "{0} has no current block", name_);
    insts_stack_.AppendToTop(inst_id);
  }

  // Returns whether the current block is statically reachable.
  auto is_current_block_reachable() -> bool {
    return id_stack_.back() != SemIR::InstBlockId::Unreachable;
  }

  // Returns a view of the contents of the top instruction block on the stack.
  auto PeekCurrentBlockContents() const -> llvm::ArrayRef<SemIR::InstId> {
    CARBON_CHECK(!empty(), "no current block");
    return insts_stack_.PeekArray();
  }

  // Prints the stack for a stack dump.
  auto PrintForStackDump(int indent, llvm::raw_ostream& output) const -> void;

  // Runs verification that the processing cleanly finished.
  auto VerifyOnFinish() const -> void {
    CARBON_CHECK(empty(), "{0} still has {1} entries", name_, id_stack_.size());
  }

  auto empty() const -> bool { return id_stack_.empty(); }

 private:
  // A name for debugging.
  llvm::StringLiteral name_;

  // The underlying SemIR::File instance. Always non-null.
  SemIR::File* sem_ir_;

  // Whether to print verbose output.
  llvm::raw_ostream* vlog_stream_;

  // The stack of block IDs. A value if allocated, `None` if no block has been
  // allocated, or `Unreachable` if this block is known to be unreachable.
  llvm::SmallVector<SemIR::InstBlockId> id_stack_;

  // The stack of insts in each block.
  ArrayStack<SemIR::InstId> insts_stack_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_INST_BLOCK_STACK_H_
