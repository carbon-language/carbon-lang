// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"

namespace Carbon {

auto SemanticsHandleArrayExpressionStart(SemanticsContext& context,
                                         ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleArrayExpressionStart");
}

auto SemanticsHandleArrayExpressionSemi(SemanticsContext& context,
                                        ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleArrayExpressionSemi");
}

auto SemanticsHandleArrayExpression(SemanticsContext& context,
                                    ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleArrayExpression");
}

}  // namespace Carbon
