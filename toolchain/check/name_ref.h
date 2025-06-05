// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_NAME_REF_H_
#define CARBON_TOOLCHAIN_CHECK_NAME_REF_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Builds a reference to the given name, which has already been resolved to
// `inst_id` within `specific_id`.
auto BuildNameRef(Context& context, SemIR::LocId loc_id, SemIR::NameId name_id,
                  SemIR::InstId inst_id, SemIR::SpecificId specific_id)
    -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_NAME_REF_H_
