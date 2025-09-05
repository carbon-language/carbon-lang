// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_THUNK_H_
#define CARBON_TOOLCHAIN_CHECK_THUNK_H_

#include "toolchain/check/context.h"
#include "toolchain/check/deferred_definition_worklist.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Given a function signature and a callee function, build a thunk that matches
// the given signature and calls the specified callee. Returns the callee
// unchanged if it can be used directly.
auto BuildThunk(Context& context, SemIR::FunctionId signature_id,
                SemIR::SpecificId signature_specific_id,
                SemIR::InstId callee_id) -> SemIR::InstId;

// Builds a call to a function that forwards a call argument list built for
// `function_id` to a call to `callee_id`, for use when building a call from a
// thunk to its target. This is like `PerformCall`, except that it takes a list
// of call arguments for `function_id`, not a syntactic argument list.
auto PerformThunkCall(Context& context, SemIR::LocId loc_id,
                      SemIR::FunctionId function_id,
                      llvm::ArrayRef<SemIR::InstId> call_arg_ids,
                      SemIR::InstId callee_id) -> SemIR::InstId;

// Builds the definition for a thunk whose definition was deferred until the end
// of the enclosing scope.
auto BuildThunkDefinition(Context& context,
                          DeferredDefinitionWorklist::DefineThunk&& task)
    -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_THUNK_H_
