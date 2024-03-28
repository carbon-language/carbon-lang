// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_FUNCTION_H_
#define CARBON_TOOLCHAIN_CHECK_FUNCTION_H_

#include "toolchain/check/context.h"
#include "toolchain/check/subst.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Checks that `new_function_id` has the same parameter types and return type as
// `prev_function_id`, applying the specified set of substitutions to the
// previous function. Prints a suitable diagnostic and returns false if not.
// Note that this doesn't include the syntactic check that's performed for
// redeclarations.
auto CheckFunctionTypeMatches(Context& context,
                              SemIR::FunctionId new_function_id,
                              SemIR::FunctionId prev_function_id,
                              Substitutions substitutions) -> bool;

// Tries to merge new_function into prev_function_id. Since new_function won't
// have a definition even if one is upcoming, set is_definition to indicate the
// planned result.
//
// If merging is successful, updates the FunctionId on new_function and returns
// true. Otherwise, returns false. Prints a diagnostic when appropriate.
auto MergeFunctionRedecl(Context& context, SemIR::LocId loc_id,
                         SemIR::Function& new_function, bool new_is_definition,
                         SemIR::FunctionId prev_function_id,
                         bool prev_is_imported) -> bool;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_FUNCTION_H_
