// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleParenExprStart(Context& /*context*/,
                          Parse::ParenExprStartId /*node_id*/) -> bool {
  // The open paren is unused.
  return true;
}

auto HandleParenExpr(Context& context, Parse::ParenExprId node_id) -> bool {
  // We re-push because the ParenExpr is valid for member expressions, whereas
  // the child expression might not be.
  context.node_stack().Push(node_id, context.node_stack().PopExpr());
  return true;
}

}  // namespace Carbon::Check
