// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/return.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

// TODO: Find a better home for this. We'll likely need it for more than just
// expression statements.
static auto HandleDiscardedExpression(Context& context, SemIR::InstId expr_id)
    -> void {
  // If we discard an initializing expression, convert it to a value or
  // reference so that it has something to initialize.
  auto expr = context.insts().Get(expr_id);
  Convert(context, expr.parse_node(), expr_id,
          {.kind = ConversionTarget::Discarded, .type_id = expr.type_id()});

  // TODO: This will eventually need to do some "do not discard" analysis.
}

auto HandleExpressionStatement(Context& context, Parse::Node /*parse_node*/)
    -> bool {
  HandleDiscardedExpression(context, context.node_stack().PopExpression());
  return true;
}

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
