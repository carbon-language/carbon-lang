// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/return.h"

namespace Carbon::Check {

auto HandleReturnStatementStart(Context& context,
                                Parse::ReturnStatementStartId node_id) -> bool {
  // No action, just a bracketing node.
  context.node_stack().Push(node_id);
  return true;
}

auto HandleReturnVarModifier(Context& context,
                             Parse::ReturnVarModifierId node_id) -> bool {
  // No action, just a bracketing node.
  context.node_stack().Push(node_id);
  return true;
}

auto HandleReturnStatement(Context& context, Parse::ReturnStatementId node_id)
    -> bool {
  switch (context.node_stack().PeekNodeKind()) {
    case Parse::NodeKind::ReturnStatementStart:
      // This is a `return;` statement.
      context.node_stack()
          .PopAndDiscardSoloNodeId<Parse::NodeKind::ReturnStatementStart>();
      BuildReturnWithNoExpr(context, node_id);
      break;

    case Parse::NodeKind::ReturnVarModifier:
      // This is a `return var;` statement.
      context.node_stack()
          .PopAndDiscardSoloNodeId<Parse::NodeKind::ReturnVarModifier>();
      context.node_stack()
          .PopAndDiscardSoloNodeId<Parse::NodeKind::ReturnStatementStart>();
      BuildReturnVar(context, node_id);
      break;

    default:
      // This is a `return <expression>;` statement.
      auto expr_id = context.node_stack().PopExpr();
      context.node_stack()
          .PopAndDiscardSoloNodeId<Parse::NodeKind::ReturnStatementStart>();
      BuildReturnWithExpr(context, node_id, expr_id);
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
