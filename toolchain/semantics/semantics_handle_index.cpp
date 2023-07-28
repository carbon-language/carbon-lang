// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "toolchain/semantics/semantics_context.h"

namespace Carbon {

auto SemanticsHandleIndexExpressionStart(SemanticsContext& context,
                                         ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleIndexExpressionStart");
}

auto SemanticsHandleIndexExpression(SemanticsContext& context,
                                    ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleIndexExpression");
}

}  // namespace Carbon
