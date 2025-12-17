// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CUSTOM_WITNESS_H_
#define CARBON_TOOLCHAIN_CHECK_CUSTOM_WITNESS_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Builds a witness that the given type implements the given interface,
// populating it with the specified set of values. Returns a corresponding
// lookup result. Produces a diagnostic and returns `None` if the specified
// values aren't suitable for the interface.
auto BuildCustomWitness(Context& context, SemIR::LocId loc_id,
                        SemIR::TypeId self_type_id,
                        SemIR::SpecificInterface specific_interface,
                        llvm::ArrayRef<SemIR::InstId> values) -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CUSTOM_WITNESS_H_
