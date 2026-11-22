// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_RETURN_H_
#define CARBON_TOOLCHAIN_CHECK_RETURN_H_

#include "toolchain/check/context.h"
#include "toolchain/parse/node_ids.h"

namespace Carbon::Check {

// Gets the function that a `return` statement in the current context would
// return from.
auto GetCurrentFunctionForReturn(Context& context) -> SemIR::Function&;

// Gets the return parameter corresponding to `function`'s `returned var`.
// Returns None if the `returned var` doesn't correspond to a return parameter
// (e.g. because it doesn't have an in-place init representation).
auto GetReturnedVarParam(Context& context, const SemIR::Function& function)
    -> SemIR::InstId;

// Checks a `returned var` binding and registers it as the current `returned
// var` in this scope.
auto RegisterReturnedVar(Context& context, Parse::NodeId returned_node,
                         Parse::NodeId type_node, SemIR::TypeId type_id,
                         SemIR::InstId bind_id, SemIR::NameId name_id) -> void;

// Checks and builds SemIR for a `return;` statement.
auto BuildReturnWithNoExpr(Context& context, SemIR::LocId loc_id) -> void;

// Checks and builds SemIR for a `return <expression>;` statement.
auto BuildReturnWithExpr(Context& context, SemIR::LocId loc_id,
                         SemIR::InstId expr_id) -> void;

// Checks and builds SemIR for a `return var;` statement.
auto BuildReturnVar(Context& context, Parse::ReturnStatementId node_id) -> void;

// Deduces the return type of the current function, which must have an
// undeduced return type, from the `return` statements in its body, and
// completes the initialization of their return values. Called at the end of the
// function body. Returns false if there are no `return` statements to deduce
// from, after diagnosing that.
auto DeduceReturnTypeAtEndOfBody(Context& context,
                                 SemIR::FunctionId function_id) -> bool;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_RETURN_H_
