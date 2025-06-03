// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_GENERIC_REGION_STACK_H_
#define CARBON_TOOLCHAIN_CHECK_GENERIC_REGION_STACK_H_

#include "common/array_stack.h"
#include "common/map.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// A map from an instruction ID representing a canonical symbolic constant to an
// instruction within an eval block of the generic that computes the specific
// value for that constant.
//
// We arbitrarily use a small size of 256 bytes for the map.
// TODO: Determine a better number based on measurements.
using ConstantsInGenericMap = Map<SemIR::InstId, SemIR::InstId, 256>;

// A stack of enclosing regions that might be declaring or defining a generic
// entity. In such a region, we track the generic constructs that are used, such
// as symbolic constants and types, and instructions that depend on a template
// parameter.
//
// We split a generic into two regions -- declaration and definition -- because
// these are in general introduced separately, and substituted into separately.
// For example, for `class C(T:! type, N:! T) { var x: T; }`, a use such as
// `C(i32, 0)*` substitutes into just the declaration, whereas a use such as
// `var x: C(i32, 0) = {.x = 0};` also substitutes into the definition.
class GenericRegionStack {
 public:
  GenericRegionStack() {
    // Reserve a large enough stack that we typically won't need to reallocate.
    constants_in_generic_stack_.reserve(4);
  }

  struct PendingGeneric {
    // The generic ID. May not have a value if no ID has been assigned yet.
    SemIR::GenericId generic_id;
    // The region of the generic that is being processed.
    SemIR::GenericInstIndex::Region region;
  };

  // Pushes a region that might be declaring or defining a generic.
  auto Push(PendingGeneric generic) -> void;

  // Pops a generic region.
  auto Pop() -> void;

  // Returns whether the stack is empty.
  auto Empty() const -> bool { return pending_generic_ids_.empty(); }

  // Sets the GenericId for the currently pending generic, once one has been
  // allocated.
  auto SetPendingGenericId(SemIR::GenericId generic_id) -> void {
    CARBON_CHECK(!pending_generic_ids_.back().generic_id.has_value(),
                 "Already have a GenericId for the pending generic");
    pending_generic_ids_.back().generic_id = generic_id;
  }

  // While held alive, instructions added as dependents or to the eval block are
  // treated as part of a single commit. If `Discard` is called, then those
  // instructions will all be dropped from the GenericRegionStack. Otherwise,
  // they will be kept.
  class [[nodiscard]] Commit {
    friend GenericRegionStack;

   public:
    ~Commit() { std::move(*this).Keep(); }

    auto Discard() && -> void {
      if (stack_) {
        stack_->dependent_inst_stack_.PopArray();
        stack_->pending_eval_block_stack_.PopArray();
        stack_->ongoing_commit_ = false;
        stack_ = nullptr;
      }
    }

    auto Keep() && -> void {
      if (stack_) {
        stack_->dependent_inst_stack_.PopAndMergeArray();
        stack_->pending_eval_block_stack_.PopAndMergeArray();
        stack_->ongoing_commit_ = false;
        stack_ = nullptr;
      }
    }

   private:
    explicit Commit(GenericRegionStack* stack) : stack_(stack) {
      if (stack_) {
        stack_->dependent_inst_stack_.PushArray();
        stack_->pending_eval_block_stack_.PushArray();
        stack_->ongoing_commit_ = true;
      }
    }

    // After move, ensure only one Commit will do cleanup.
    Commit(Commit&& rhs) noexcept
        : stack_(std::exchange(rhs.stack_, nullptr)) {}
    auto operator=(Commit&& rhs) noexcept -> Commit& {
      stack_ = std::exchange(rhs.stack_, nullptr);
      return *this;
    }

    GenericRegionStack* stack_;
  };

  // Begin a commit, where instructions may be created but thrown away. At the
  // end of the commit, the caller may choose to keep all instructions added as
  // dependent or to the eval block, or to discard them all. If they are
  // discarded, they must not be referenced afterward.
  auto BeginCommit() -> Commit { return Commit(Empty() ? nullptr : this); }

  // Adds an instruction to the list of instructions whose type or value depends
  // on something in the current pending generic.
  auto AddDependentInst(SemIR::InstId inst_id) -> void {
    CARBON_CHECK(!Empty());
    dependent_inst_stack_.AppendToTop(inst_id);
  }

  // Adds an instruction to the eval block for the current pending generic.
  auto AddInstToEvalBlock(SemIR::InstId inst_id) -> void {
    CARBON_CHECK(!Empty());
    pending_eval_block_stack_.AppendToTop(inst_id);
  }

  // Returns the current pending generic.
  auto PeekPendingGeneric() const -> PendingGeneric {
    CARBON_CHECK(!Empty());
    return pending_generic_ids_.back();
  }

  // Returns the list of dependent instructions in the current generic region.
  auto PeekDependentInsts() -> llvm::ArrayRef<SemIR::InstId> {
    CARBON_CHECK(!Empty());
    return dependent_inst_stack_.PeekArray();
  }

  // Returns the contents of the eval block for the current generic region.
  auto PeekEvalBlock() -> llvm::ArrayRef<SemIR::InstId> {
    CARBON_CHECK(!Empty());
    return pending_eval_block_stack_.PeekArray();
  }

  // Returns the mapping from abstract constant instructions to eval block
  // instructions for the current generic.
  auto PeekConstantsInGenericMap() -> ConstantsInGenericMap& {
    CARBON_CHECK(!Empty());
    return constants_in_generic_stack_.back();
  }

  // Runs verification that the processing cleanly finished.
  auto VerifyOnFinish() const -> void {
    CARBON_CHECK(pending_generic_ids_.empty(),
                 "pending_generic_ids_ still has {0} entries",
                 pending_generic_ids_.size());
  }

 private:
  // The IDs of pending generics.
  llvm::SmallVector<PendingGeneric> pending_generic_ids_;

  // Contents of eval blocks for pending generics.
  ArrayStack<SemIR::InstId> pending_eval_block_stack_;

  // Instructions that depend on the current generic.
  ArrayStack<SemIR::InstId> dependent_inst_stack_;

  // Mapping from constant InstIds to the corresponding InstIds in the eval
  // blocks for each enclosing generic. We reserve this to a suitable size in
  // the constructor.
  llvm::SmallVector<ConstantsInGenericMap, 0> constants_in_generic_stack_;

  bool ongoing_commit_ = false;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_GENERIC_REGION_STACK_H_
