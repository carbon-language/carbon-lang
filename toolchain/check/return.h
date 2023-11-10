// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_RETURN_H_
#define CARBON_TOOLCHAIN_CHECK_RETURN_H_

#include "toolchain/check/context.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

// Checks a `returned var` binding and returns the location of the return
// value storage that the name should bind to.
auto CheckReturnedVar(Context& context, Parse::Node returned_node,
                      Parse::Node name_node, SemIR::NameId name_id,
                      Parse::Node type_node, SemIR::TypeId type_id)
    -> SemIR::InstId;

// Registers the given binding as the current `returned var` in this scope.
auto RegisterReturnedVar(Context& context, SemIR::InstId bind_id) -> void;

// Checks and builds SemIR for a `return;` statement.
auto BuildReturnWithNoExpr(Context& context, Parse::Node parse_node) -> void;

// Checks and builds SemIR for a `return <expression>;` statement.
auto BuildReturnWithExpr(Context& context, Parse::Node parse_node,
                         SemIR::InstId expr_id) -> void;

// Checks and builds SemIR for a `return var;` statement.
auto BuildReturnVar(Context& context, Parse::Node parse_node) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_RETURN_H_
