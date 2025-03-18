// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_DEDUCE_H_
#define CARBON_TOOLCHAIN_CHECK_DEDUCE_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Deduces the generic arguments to use in a call to a generic.
auto DeduceGenericCallArguments(
    Context& context, SemIR::LocId loc_id, SemIR::GenericId generic_id,
    SemIR::SpecificId enclosing_specific_id, SemIR::InstId self_type_id,
    SemIR::InstBlockId implicit_params_id, SemIR::InstBlockId params_id,
    SemIR::InstId self_id, llvm::ArrayRef<SemIR::InstId> arg_ids)
    -> SemIR::SpecificId;

// Data from the `Impl` that is used by deduce.
//
// We don't use a reference to an `Impl` as deduction can invalidate the
// reference by causing impl declarations to be imported from `Core` during
// conversion.
struct DeduceImpl {
  SemIR::InstId self_id;
  SemIR::GenericId generic_id;
  SemIR::SpecificId specific_id;
};

// Deduces the impl arguments to use in a use of a parameterized impl. Returns
// `None` if deduction fails.
auto DeduceImplArguments(Context& context, SemIR::LocId loc_id, DeduceImpl impl,
                         SemIR::ConstantId self_id,
                         SemIR::SpecificId constraint_specific_id)
    -> SemIR::SpecificId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_DEDUCE_H_
