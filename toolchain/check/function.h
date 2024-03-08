// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_FUNCTION_H_
#define CARBON_TOOLCHAIN_CHECK_FUNCTION_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/function.h"

namespace Carbon::Check {

// Checks that `new_function_id` does not differ from `prev_function_id`.
// Prints a suitable diagnostic and returns false if not.
auto CheckFunctionRedecl(Context& context, SemIR::FunctionId new_function_id,
                         SemIR::FunctionId prev_function_id) -> bool;

// Tries to merge new_function into prev_function_id. Since new_function won't
// have a definition even if one is upcoming, set is_definition to indicate the
// planned result.
//
// If merging is successful, updates the FunctionId on new_function and returns
// true. Otherwise, returns false. Prints a diagnostic when appropriate.
auto MergeFunctionRedecl(Context& context, Parse::NodeId node_id,
                         SemIR::Function& new_function,
                         SemIR::FunctionId prev_function_id, bool is_definition)
    -> bool;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_FUNCTION_H_
