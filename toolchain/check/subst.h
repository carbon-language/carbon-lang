// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_SUBST_H_
#define CARBON_TOOLCHAIN_CHECK_SUBST_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// A substitution that is being performed.
struct Substitution {
  // The index of a `BindSymbolicName` instruction that is being replaced.
  SemIR::CompileTimeBindIndex bind_id;
  // The replacement constant value to substitute.
  SemIR::ConstantId replacement_id;
};

using Substitutions = llvm::ArrayRef<Substitution>;

// A function that performs any needed substitution into an instruction, as part
// of substituting into a symbolic instruction sequence. The instruction ID
// should be updated as necessary to represent the new instruction. If the
// function returns true, the instruction is treated as fully-substituted; if it
// returns false, the instruction will be decomposed and substitution will be
// performed recursively into its operands.
using SubstInstFn = llvm::function_ref<auto(SemIR::InstId& inst_id)->bool>;

// A function that rebuilds an instruction after substitution. `orig_inst_id` is
// the instruction prior to substitution, and `new_inst` is the substituted
// instruction. Returns the new instruction ID to use to refer to `new_inst`.
using SubstRebuildFn = llvm::function_ref<
    auto(SemIR::InstId orig_inst_id, SemIR::Inst new_inst)->SemIR::InstId>;

// Performs substitution into `inst_id` and its operands recursively.
auto SubstInst(Context& context, SemIR::InstId inst_id, SubstInstFn subst_fn,
               SubstRebuildFn rebuild_fn) -> SemIR::InstId;

// Replaces the `BindSymbolicName` instruction `bind_id` with `replacement_id`
// throughout the constant `const_id`, and returns the substituted value.
auto SubstConstant(Context& context, SemIR::ConstantId const_id,
                   Substitutions substitutions) -> SemIR::ConstantId;

// Replaces the `BindSymbolicName` instruction `bind_id` with `replacement_id`
// throughout the type `type_id`, and returns the substituted value.
auto SubstType(Context& context, SemIR::TypeId type_id,
               Substitutions substitutions) -> SemIR::TypeId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_SUBST_H_
