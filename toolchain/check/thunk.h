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

// Builds the definition for a thunk whose definition was deferred until the end
// of the enclosing scope.
auto BuildThunkDefinition(Context& context,
                          DeferredDefinitionWorklist::DefineThunk&& task)
    -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_THUNK_H_
