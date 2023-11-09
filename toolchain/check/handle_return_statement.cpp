// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/return.h"

namespace Carbon::Check {

auto HandleReturnStatementStart(Context& context, Parse::Node parse_node)
    -> bool {
  // No action, just a bracketing node.
  context.node_stack().Push(parse_node);
  return true;
}

auto HandleReturnVarSpecifier(Context& context, Parse::Node parse_node)
    -> bool {
  // No action, just a bracketing node.
  context.node_stack().Push(parse_node);
  return true;
}

auto HandleReturnStatement(Context& context, Parse::Node parse_node) -> bool {
  switch (
      context.parse_tree().node_kind(context.node_stack().PeekParseNode())) {
    case Parse::NodeKind::ReturnStatementStart:
      // This is a `return;` statement.
      context.node_stack()
          .PopAndDiscardSoloParseNode<Parse::NodeKind::ReturnStatementStart>();
      BuildReturnWithNoExpression(context, parse_node);
      break;

    case Parse::NodeKind::ReturnVarSpecifier:
      // This is a `return var;` statement.
      context.node_stack()
          .PopAndDiscardSoloParseNode<Parse::NodeKind::ReturnVarSpecifier>();
      context.node_stack()
          .PopAndDiscardSoloParseNode<Parse::NodeKind::ReturnStatementStart>();
      BuildReturnVar(context, parse_node);
      break;

    default:
      // This is a `return <expression>;` statement.
      auto expr_id = context.node_stack().PopExpression();
      context.node_stack()
          .PopAndDiscardSoloParseNode<Parse::NodeKind::ReturnStatementStart>();
      BuildReturnWithExpression(context, parse_node, expr_id);
      break;
  }

  // Switch to a new, unreachable, empty instruction block. This typically won't
  // contain any semantics IR, but it can do if there are statements following
  // the `return` statement.
  context.inst_block_stack().Pop();
  context.inst_block_stack().PushUnreachable();
  return true;
}

}  // namespace Carbon::Check
