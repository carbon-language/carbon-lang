// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_THUNK_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_THUNK_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Returns whether the given C++ imported function requires a C++ thunk to be
// used to call it. A C++ thunk is required for functions that use any type
// except void, pointer types and signed 32-bit and 64-bit integers.
auto IsCppThunkRequired(Context& context, const SemIR::Function& function)
    -> bool;

// Given a function signature and a callee function, builds a C++ thunk with
// simple ABI (pointers, i32 and i64 types) that calls the specified callee.
// Assumes `IsCppThunkRequired()` return true for `callee_function`. Returns
// `nullptr` on failure.
auto BuildCppThunk(Context& context, const SemIR::Function& callee_function)
    -> clang::FunctionDecl*;

// Builds a call to a thunk function that forwards a call argument list built
// for `callee_function_id` to a call to `thunk_callee_id`, for use when
// building a call from a C++ thunk to its target. This is like `PerformCall`,
// except that it takes a list of call arguments for `callee_function_id`, not a
// syntactic argument list.
auto PerformCppThunkCall(Context& context, SemIR::LocId loc_id,
                         SemIR::FunctionId callee_function_id,
                         llvm::ArrayRef<SemIR::InstId> callee_arg_ids,
                         SemIR::InstId thunk_callee_id) -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_THUNK_H_
