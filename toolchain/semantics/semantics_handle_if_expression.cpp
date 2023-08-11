// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"

namespace Carbon {

auto SemanticsHandleIfExpressionIf(SemanticsContext& context,
                                   ParseTree::Node if_node) -> bool {
  auto cond_value_id = context.node_stack().PopExpression();

  context.node_stack().Push(if_node);

  // Convert the condition to `bool`, and branch on it.
  cond_value_id = context.ConvertToBoolValue(if_node, cond_value_id);
  auto then_block_id =
      context.AddDominatedBlockAndBranchIf(if_node, cond_value_id);
  auto else_block_id = context.AddDominatedBlockAndBranch(if_node);

  // Push the `else` block and `then` block, and start emitting the `then`.
  context.node_block_stack().Pop();
  context.node_block_stack().Push(else_block_id);
  context.node_block_stack().Push(then_block_id);
  context.AddCurrentCodeBlockToFunction();
  return true;
}

auto SemanticsHandleIfExpressionThen(SemanticsContext& context,
                                     ParseTree::Node then_node) -> bool {
  // Convert the first operand to a value.
  auto [then_value_node, then_value_id] =
      context.node_stack().PopExpressionWithParseNode();
  context.node_stack().Push(then_value_node,
                            context.ConvertToValueExpression(then_value_id));

  context.node_stack().Push(then_node, context.node_block_stack().Pop());
  context.AddCurrentCodeBlockToFunction();
  return true;
}

auto SemanticsHandleIfExpressionElse(SemanticsContext& context,
                                     ParseTree::Node else_node) -> bool {
  auto else_value_id = context.node_stack().PopExpression();
  auto [then_node, then_end_block_id] =
      context.node_stack().PopWithParseNode<ParseNodeKind::IfExpressionThen>();
  auto then_value_id = context.node_stack().PopExpression();
  auto if_node =
      context.node_stack().PopForSoloParseNode<ParseNodeKind::IfExpressionIf>();

  // Convert the `else` value to the `then` value's type, and finish the `else`
  // block.
  // TODO: Find a common type, and convert both operands to it instead.
  auto result_type_id = context.semantics_ir().GetNode(then_value_id).type_id();
  else_value_id =
      context.ConvertToValueOfType(else_node, else_value_id, result_type_id);
  auto else_end_block_id = context.node_block_stack().Pop();

  // Create a resumption block and branches to it.
  auto chosen_value_id = context.AddConvergenceBlockWithArgAndPush(
      if_node,
      {{then_end_block_id, then_value_id}, {else_end_block_id, else_value_id}});
  context.AddCurrentCodeBlockToFunction();

  // Push the result value.
  context.node_stack().Push(else_node, chosen_value_id);
  return true;
}

}  // namespace Carbon
