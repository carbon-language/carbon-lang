// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_SUBST_H_
#define CARBON_TOOLCHAIN_CHECK_SUBST_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Callbacks used by SubstInst to recursively substitute into and rebuild an
// instruction.
class SubstInstCallbacks {
 public:
  explicit SubstInstCallbacks(Context* context) : context_(context) {}

  auto context() const -> Context& { return *context_; }

  // How further substitution should or should not be applied to the instruction
  // after Subst is done.
  //
  // Rebuild or ReuseUnchaged will always be called when SubstAgain or
  // SubstOperands is returned, after processing anything inside the instruction
  // after Subst.
  enum SubstResult {
    // Don't substitute into the operands of the instruction.
    FullySubstituted,
    // Attempt to substitute into the operands of the instruction.
    SubstOperands,
    // Attempt to substitute again on the resulting instruction, acting like
    // recursion on the instruction itself.
    SubstAgain,
    // Attempt to substitute into the operands of the instruction. If the InstId
    // returned from Rebuild or ReuseUnchanged differs from the input (typically
    // because some operand in the instruction changed), then the new
    // instruction will be given to `Subst` again afterward. This allows for the
    // uncommon case of substituting from the inside out.
    SubstOperandsAndRetry,
  };

  // Performs any needed substitution into an instruction. The instruction ID
  // should be updated as necessary to represent the new instruction.
  //
  // Return FullySubstituted if the resulting instruction ID is
  // fully-substituted. Return SubstOperands if substitution may be needed into
  // operands of the instruction, or SubstAgain if the replaced instruction
  // itself should have substitution applied to it again. Return
  // SubstOperandsAndRetry to recurse on the instructions operands and then
  // substitute the resulting instruction afterward, if the instruction is
  // replaced by a new one (typically due to Rebuild when the operands changed).
  //
  // When SubstOperands, SubstAgain, or SubstOperandsAndRetry is returned, it
  // results in a call back to Rebuild or ReuseUnchanged when that instruction's
  // substitution step is complete.
  virtual auto Subst(SemIR::InstId& inst_id) -> SubstResult = 0;

  // Rebuilds the type of an instruction from the substituted type instruction.
  // By default this builds the unattached type described by the given type ID.
  virtual auto RebuildType(SemIR::TypeInstId type_inst_id) const
      -> SemIR::TypeId;

  // Rebuilds an instruction whose operands were changed by substitution.
  // `orig_inst_id` is the instruction prior to substitution, and `new_inst` is
  // the substituted instruction. Returns the new instruction ID to use to refer
  // to `new_inst`.
  virtual auto Rebuild(SemIR::InstId orig_inst_id, SemIR::Inst new_inst)
      -> SemIR::InstId = 0;

  // Performs any work needed when no substitutions were performed into an
  // instruction for which `Subst` returned `false`. Provides an opportunity to
  // perform any necessary updates to the instruction beyond updating its
  // operands. Returns the new instruction ID to use to refer to `orig_inst_id`.
  virtual auto ReuseUnchanged(SemIR::InstId orig_inst_id) -> SemIR::InstId {
    return orig_inst_id;
  }

  // Builds a new constant by evaluating `new_inst`, and returns its `InstId`.
  //
  // This can be used to implement `Rebuild` in straightforward cases.
  auto RebuildNewInst(SemIR::LocId loc_id, SemIR::Inst new_inst) const
      -> SemIR::InstId;

  template <typename InstT>
  auto RebuildNewInst(SemIR::LocId loc_id, InstT new_inst) const
      -> SemIR::InstId {
    return RebuildNewInst(loc_id, static_cast<SemIR::Inst>(new_inst));
  }

 private:
  Context* context_;
};

// Performs substitution into `inst_id` and its operands recursively, using
// `callbacks` to process each instruction. For each instruction encountered,
// calls `Subst` to perform substitution on that instruction.
auto SubstInst(Context& context, SemIR::InstId inst_id,
               SubstInstCallbacks& callbacks) -> SemIR::InstId;
auto SubstInst(Context& context, SemIR::TypeInstId inst_id,
               SubstInstCallbacks& callbacks) -> SemIR::TypeInstId;

// A substitution that is being performed.
struct Substitution {
  // The index of a `BindSymbolicName` instruction that is being replaced.
  SemIR::CompileTimeBindIndex bind_id;
  // The replacement constant value to substitute.
  SemIR::ConstantId replacement_id;
};

using Substitutions = llvm::ArrayRef<Substitution>;

// Replaces the `BindSymbolicName` instruction `bind_id` with `replacement_id`
// throughout the constant `const_id`, and returns the substituted value.
auto SubstConstant(Context& context, SemIR::LocId loc_id,
                   SemIR::ConstantId const_id, Substitutions substitutions)
    -> SemIR::ConstantId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_SUBST_H_
