// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleBreakStatement(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleBreakStatement");
}

auto HandleBreakStatementStart(Context& context, Parse::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleBreakStatementStart");
}

auto HandleContinueStatement(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleContinueStatement");
}

auto HandleContinueStatementStart(Context& context, Parse::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleContinueStatementStart");
}

auto HandleForHeader(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleForHeader");
}

auto HandleForHeaderStart(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleForHeaderStart");
}

auto HandleForIn(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleForIn");
}

auto HandleForStatement(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleForStatement");
}

auto HandleWhileCondition(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleWhileCondition");
}

auto HandleWhileConditionStart(Context& context, Parse::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleWhileConditionStart");
}

auto HandleWhileStatement(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleWhileStatement");
}

}  // namespace Carbon::Check
