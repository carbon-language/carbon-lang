// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_FUNCTION_H_
#define CARBON_TOOLCHAIN_CHECK_FUNCTION_H_

#include "toolchain/check/context.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/subst.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// State saved for a function definition that has been suspended after
// processing its declaration and before processing its body. This is used for
// inline method handling.
struct SuspendedFunction {
  // The function that was declared.
  SemIR::FunctionId function_id;
  // The instruction ID of the FunctionDecl instruction.
  SemIR::InstId decl_id;
  // The declaration name information of the function. This includes the scope
  // information, such as parameter names.
  DeclNameStack::SuspendedName saved_name_state;
};

// Checks that `new_function` has the same parameter types and return type as
// `prev_function`, applying the specified set of substitutions to the
// previous function. Prints a suitable diagnostic and returns false if not.
// Note that this doesn't include the syntactic check that's performed for
// redeclarations.
auto CheckFunctionTypeMatches(Context& context,
                              const SemIR::Function& new_function,
                              const SemIR::Function& prev_function,
                              Substitutions substitutions) -> bool;

// Checks that the return type of the specified function is complete, issuing an
// error if not. This computes the return slot usage for the function if
// necessary.
auto CheckFunctionReturnType(Context& context, SemIRLoc loc,
                             SemIR::Function& function) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_FUNCTION_H_
