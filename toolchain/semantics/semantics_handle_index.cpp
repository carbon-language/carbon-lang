// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "toolchain/semantics/semantics_context.h"

namespace Carbon {

auto SemanticsHandleIndexExpressionStart(SemanticsContext& /*context*/,
                                         ParseTree::Node /*parse_node*/)
    -> bool {
  // TODO
  return true;
}

auto SemanticsHandleIndexExpression(SemanticsContext& /*context*/,
                                    ParseTree::Node /*parse_node*/) -> bool {
  // TODO
  return true;
}

}  // namespace Carbon
