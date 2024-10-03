// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_DEDUCE_H_
#define CARBON_TOOLCHAIN_CHECK_DEDUCE_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Deduces the generic arguments to use in a call to a generic.
auto DeduceGenericCallArguments(Context& context, SemIR::LocId loc_id,
                                SemIR::GenericId generic_id,
                                SemIR::SpecificId enclosing_specific_id,
                                SemIR::InstBlockId implicit_params_id,
                                SemIR::InstBlockId params_id,
                                SemIR::InstId self_id,
                                llvm::ArrayRef<SemIR::InstId> arg_ids)
    -> SemIR::SpecificId;

// Deduces the impl arguments to use in a use of a parameterized impl. Returns
// `Invalid` if deduction fails.
auto DeduceImplArguments(Context& context, const SemIR::Impl& impl,
                         SemIR::ConstantId self_id,
                         SemIR::ConstantId constraint_id) -> SemIR::SpecificId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_DEDUCE_H_
