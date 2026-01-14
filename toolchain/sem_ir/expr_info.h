// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_EXPR_INFO_H_
#define CARBON_TOOLCHAIN_SEM_IR_EXPR_INFO_H_

#include <cstdint>

#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_kind.h"

namespace Carbon::SemIR {

// Returns the expression category for an instruction.
auto GetExprCategory(const File& file, InstId inst_id) -> ExprCategory;

// Returns whether the given expression category is for a reference expression.
inline auto IsRefCategory(ExprCategory cat) -> bool {
  return cat == ExprCategory::DurableRef || cat == ExprCategory::EphemeralRef;
}

// Given a primitive-form initializing expression, find its return slot
// argument. Returns `None` if there is no return slot, because the
// initialization is not performed in place.
auto FindReturnSlotArgForInitializer(const File& sem_ir, InstId init_id)
    -> InstId;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_EXPR_INFO_H_
