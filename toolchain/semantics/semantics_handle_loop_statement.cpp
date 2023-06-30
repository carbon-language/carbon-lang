// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

auto SemanticsHandleBreakStatement(SemanticsContext& context,
                                   ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleBreakStatement");
}

auto SemanticsHandleBreakStatementStart(SemanticsContext& context,
                                        ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleBreakStatementStart");
}

auto SemanticsHandleContinueStatement(SemanticsContext& context,
                                      ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleContinueStatement");
}

auto SemanticsHandleContinueStatementStart(SemanticsContext& context,
                                           ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleContinueStatementStart");
}

auto SemanticsHandleForHeader(SemanticsContext& context,
                              ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleForHeader");
}

auto SemanticsHandleForHeaderStart(SemanticsContext& context,
                                   ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleForHeaderStart");
}

auto SemanticsHandleForIn(SemanticsContext& context, ParseTree::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleForIn");
}

auto SemanticsHandleForStatement(SemanticsContext& context,
                                 ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleForStatement");
}

auto SemanticsHandleWhileCondition(SemanticsContext& context,
                                   ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleWhileCondition");
}

auto SemanticsHandleWhileConditionStart(SemanticsContext& context,
                                        ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleWhileConditionStart");
}

auto SemanticsHandleWhileStatement(SemanticsContext& context,
                                   ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleWhileStatement");
}

}  // namespace Carbon
