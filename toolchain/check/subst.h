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
  // Performs any needed substitution into an instruction. The instruction ID
  // should be updated as necessary to represent the new instruction. Returns
  // true if the resulting instruction ID is fully-substituted, or false if
  // substitution may be needed into operands of the instruction.
  virtual auto Subst(SemIR::InstId& inst_id) const -> bool = 0;

  // Rebuilds an instruction whose operands were changed by substitution.
  // `orig_inst_id` is the instruction prior to substitution, and `new_inst` is
  // the substituted instruction. Returns the new instruction ID to use to refer
  // to `new_inst`.
  virtual auto Rebuild(SemIR::InstId orig_inst_id, SemIR::Inst new_inst) const
      -> SemIR::InstId = 0;

  // Performs any work needed when no substitutions were performed into an
  // instruction for which `Subst` returned `false`. Provides an opportunity to
  // perform any necessary updates to the instruction beyond updating its
  // operands. Returns the new instruction ID to use to refer to `orig_inst_id`.
  virtual auto ReuseUnchanged(SemIR::InstId orig_inst_id) const
      -> SemIR::InstId {
    return orig_inst_id;
  }
};

// Performs substitution into `inst_id` and its operands recursively, using
// `callbacks` to process each instruction. For each instruction encountered,
// calls `Subst` to perform substitution on that instruction.
//
// If `Subst` returns false, the instruction is decomposed into its operands,
// which are substituted recursively, and if any of them change then `Rebuild`
// is used to build a new instruction with the substituted operands.
auto SubstInst(Context& context, SemIR::InstId inst_id,
               const SubstInstCallbacks& callbacks) -> SemIR::InstId;

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
auto SubstConstant(Context& context, SemIR::ConstantId const_id,
                   Substitutions substitutions) -> SemIR::ConstantId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_SUBST_H_
