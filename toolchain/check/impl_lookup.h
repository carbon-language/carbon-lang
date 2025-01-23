// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_IMPL_LOOKUP_H_
#define CARBON_TOOLCHAIN_CHECK_IMPL_LOOKUP_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Looks up the witness to use for a particular type and interface. Returns the
// witness, or `InstId::None` if the type is not known to implement the
// interface.
auto LookupImplWitness(Context& context, SemIR::LocId loc_id,
                       SemIR::ConstantId type_const_id,
                       SemIR::ConstantId interface_const_id) -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_IMPL_LOOKUP_H_
