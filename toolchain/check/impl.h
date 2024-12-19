// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_IMPL_H_
#define CARBON_TOOLCHAIN_CHECK_IMPL_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Returns the initial witness value for a new impl declaration.
auto ImplWitnessForDeclaration(Context& context, const SemIR::Impl& impl)
    -> SemIR::InstId;

// Builds and returns a witness for the impl `impl_id`.
auto BuildImplWitness(Context& context, const SemIR::Impl& impl)
    -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_IMPL_H_
