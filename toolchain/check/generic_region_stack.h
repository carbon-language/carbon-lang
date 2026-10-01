// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_GENERIC_REGION_STACK_H_
#define CARBON_TOOLCHAIN_CHECK_GENERIC_REGION_STACK_H_

#include "common/array_stack.h"
#include "common/map.h"
#include "llvm/ADT/STLExtras.h"
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
// For example, for `class C(T: type, N: T) { var x: T; }`, a use such as
// `C(i32, 0)*` substitutes into just the declaration, whereas a use such as
// `var x: C(i32, 0) = {.x = 0};` also substitutes into the definition.
class GenericRegionStack {
 public:
  explicit GenericRegionStack(llvm::raw_ostream* vlog_stream)
      : vlog_stream_(vlog_stream) {
    // Reserve a large enough stack that we typically won't need to reallocate.
    constants_in_generic_stack_.reserve(4);
  }

  // Information about the pack expansion whose body is a generic region.
  struct PackExpansionInfo {
    // Sentinel values for `arity`.
    static constexpr int32_t UnknownArity = -1;
    static constexpr int32_t ErrorArity = -2;

    // The variadic index binding. None if this region is not the body of a
    // pack expansion.
    SemIR::InstId index_id = SemIR::InstId::None;
    // The arity of the pack expansion, if known.
    int32_t arity = UnknownArity;
    // The operand of the `expand` expression that determined the arity, if
    // any.
    SemIR::InstId arity_source_id = SemIR::InstId::None;
  };

  struct PendingGeneric {
    // The generic ID. May not have a value if no ID has been assigned yet.
    SemIR::GenericId generic_id;
    // The region of the generic that is being processed.
    SemIR::GenericInstIndex::Region region;
    // If this region is the body of a pack expansion, information about that
    // pack expansion.
    PackExpansionInfo pack_expansion = {};
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

  // Marks the current generic region as being the body of a pack expansion
  // with the given variadic index binding.
  auto SetPackExpansionIndex(SemIR::InstId index_id) -> void {
    CARBON_CHECK(!Empty());
    pending_generic_ids_.back().pack_expansion = {.index_id = index_id};
  }

  // Returns the pack expansion information for the current generic region.
  // Returns null if there is no current generic region or it is not the body
  // of a pack expansion.
  // TODO: Consider looking through enclosing regions for the innermost pack
  // expansion.
  auto PeekPackExpansion() -> PackExpansionInfo* {
    if (Empty() ||
        !pending_generic_ids_.back().pack_expansion.index_id.has_value()) {
      return nullptr;
    }
    return &pending_generic_ids_.back().pack_expansion;
  }

  // Returns whether any enclosing generic region is the body of a pack
  // expansion.
  auto IsInPackExpansion() const -> bool {
    return llvm::any_of(pending_generic_ids_,
                        [](const PendingGeneric& pending) {
                          return pending.pack_expansion.index_id.has_value();
                        });
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
  // Whether to print verbose output.
  llvm::raw_ostream* vlog_stream_;

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
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_GENERIC_REGION_STACK_H_
