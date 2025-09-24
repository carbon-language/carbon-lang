// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_OPERATORS_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_OPERATORS_H_

#include "toolchain/check/context.h"
#include "toolchain/check/operator.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Looks up the given operator in the Clang AST generated when importing C++
// code using argument dependent lookup (ADL) and return overload set
// instruction.
auto LookupCppOperator(Context& context, SemIR::LocId loc_id, Operator op,
                       llvm::ArrayRef<SemIR::InstId> arg_ids) -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_OPERATORS_H_
