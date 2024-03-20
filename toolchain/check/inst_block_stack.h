// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_INST_BLOCK_STACK_H_
#define CARBON_TOOLCHAIN_CHECK_INST_BLOCK_STACK_H_

#include "llvm/ADT/SmallVector.h"
#include "toolchain/sem_ir/file.h"

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

  // Pushes a new instruction block. It will be invalid unless PeekOrAdd is
  // called in order to support lazy allocation.
  auto Push() -> void { Push(SemIR::InstBlockId::Invalid); }

  // Pushes the `GlobalInit` inst block onto the stack, this block is handled
  // separately from the rest.
  // This method shall be used in conjunction with `PopGlobalInit` method to
  // allow emitting initialization instructions to `GlobalInit` block from
  // separate parts of the tree, accumulating them all in one block.
  auto PushGlobalInit() -> void;

  // Pushes a new unreachable code block.
  auto PushUnreachable() -> void { Push(SemIR::InstBlockId::Unreachable); }

  // Returns the ID of the top instruction block, allocating one if necessary.
  // If `depth` is specified, returns the instruction at `depth` levels from the
  // top of the stack instead of the top block, where the top block is at depth
  // 0.
  auto PeekOrAdd(int depth = 0) -> SemIR::InstBlockId;

  // Pops the top instruction block. This will always return a valid instruction
  // block; SemIR::InstBlockId::Empty is returned if one wasn't allocated.
  auto Pop() -> SemIR::InstBlockId;

  // Pops the top instruction block, and discards it if it hasn't had an ID
  // allocated.
  auto PopAndDiscard() -> void;

  // Pops the `GlobalInit` inst block from the stack without finalizing it.
  // `Pop` should be called at the end of the check phase, while `GlobalInit`
  // is pushed, to finalize the block.
  auto PopGlobalInit() -> void;

  // Adds the given instruction ID to the block at the top of the stack.
  auto AddInstId(SemIR::InstId inst_id) -> void {
    CARBON_CHECK(!empty()) << "no current block";
    stack_[size_ - 1].content.push_back(inst_id);
  }

  // Adds the given instruction ID to the block at the bottom of the stack.
  //
  // TODO: We shouldn't need to do this.
  auto AddInstIdToFileBlock(SemIR::InstId inst_id) -> void {
    CARBON_CHECK(!empty()) << "no current block";
    stack_[0].content.push_back(inst_id);
  }

  // Returns whether the current block is statically reachable.
  auto is_current_block_reachable() -> bool {
    return size_ != 0 &&
           stack_[size_ - 1].id != SemIR::InstBlockId::Unreachable;
  }

  // Returns a view of the contents of the top instruction block on the stack.
  auto PeekCurrentBlockContents() -> llvm::ArrayRef<SemIR::InstId> {
    CARBON_CHECK(!empty()) << "no current block";
    return stack_[size_ - 1].content;
  }

  // Prints the stack for a stack dump.
  auto PrintForStackDump(llvm::raw_ostream& output) const -> void;

  // Runs verification that the processing cleanly finished.
  auto VerifyOnFinish() const -> void { CARBON_CHECK(empty()) << size_; }

  auto empty() const -> bool { return size_ == 0; }

 private:
  struct StackEntry {
    // Preallocate an arbitrary size for the stack entries.
    // TODO: Perform measurements to pick a good starting size to avoid
    // reallocation.
    StackEntry() { content.reserve(32); }

    auto Reset(SemIR::InstBlockId new_id) {
      id = new_id;
      content.clear();
    }

    // The block ID, if one has been allocated, Invalid if no block has been
    // allocated, or Unreachable if this block is known to be unreachable.
    SemIR::InstBlockId id = SemIR::InstBlockId::Invalid;

    // The content of the block. Stored as a vector rather than as a SmallVector
    // to reduce the cost of resizing `stack_` and performing swaps.
    std::vector<SemIR::InstId> content;
  };

  // A name for debugging.
  llvm::StringLiteral name_;

  // The underlying SemIR::File instance. Always non-null.
  SemIR::File* sem_ir_;

  // Whether to print verbose output.
  llvm::raw_ostream* vlog_stream_;

  std::vector<SemIR::InstId> init_block_;

  // The actual stack.
  llvm::SmallVector<StackEntry> stack_;

  // The size of the stack. Entries after this in `stack_` are kept around so
  // that we can reuse the allocated buffer for their content.
  int size_ = 0;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_INST_BLOCK_STACK_H_
