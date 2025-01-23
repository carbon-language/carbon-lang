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

// Gets the return slot of the function that lexically encloses the current
// location.
auto GetCurrentReturnSlot(Context& context) -> SemIR::InstId;

// Checks a `returned var` binding and registers it as the current `returned
// var` in this scope.
auto RegisterReturnedVar(Context& context, Parse::NodeId returned_node,
                         Parse::NodeId type_node, SemIR::TypeId type_id,
                         SemIR::InstId bind_id) -> void;

// Checks and builds SemIR for a `return;` statement.
auto BuildReturnWithNoExpr(Context& context, Parse::ReturnStatementId node_id)
    -> void;

// Checks and builds SemIR for a `return <expression>;` statement.
auto BuildReturnWithExpr(Context& context, Parse::ReturnStatementId node_id,
                         SemIR::InstId expr_id) -> void;

// Checks and builds SemIR for a `return var;` statement.
auto BuildReturnVar(Context& context, Parse::ReturnStatementId node_id) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_RETURN_H_
