// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/handle.h"
#include "toolchain/parse/typed_nodes.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::ParenExprStartId node_id)
    -> bool {
  // Push the start to help track nesting.
  context.node_stack().Push(node_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::ParenExprId node_id) -> bool {
  auto expr = context.node_stack().PopExpr();
  context.node_stack().PopForSoloNodeId<Parse::NodeKind::ParenExprStart>();
  // We push with the ParenExpr node because it's valid for member expressions,
  // whereas the child expression might not be.
  context.node_stack().Push(node_id, expr);
  return true;
}

}  // namespace Carbon::Check
