// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_IMPL_H_
#define CARBON_TOOLCHAIN_CHECK_IMPL_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Returns the initial witness value for a new `impl` declaration.
//
// `has_definition` is whether this declaration is immediately followed by the
// opening of the definition.
auto ImplWitnessForDeclaration(Context& context, const SemIR::Impl& impl,
                               bool has_definition) -> SemIR::InstId;

// Update `impl`'s witness at the start of a definition.
auto ImplWitnessStartDefinition(Context& context, SemIR::Impl& impl) -> void;

// Adds the function members to the witness for `impl`.
auto FinishImplWitness(Context& context, SemIR::Impl& impl) -> void;

// Sets all unset members of the witness for `impl` to the error instruction.
auto FillImplWitnessWithErrors(Context& context, SemIR::Impl& impl) -> void;

// Returns whether the impl is either `final` explicitly, or implicitly due to
// being concrete.
auto IsImplEffectivelyFinal(Context& context, const SemIR::Impl& impl) -> bool;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_IMPL_H_
