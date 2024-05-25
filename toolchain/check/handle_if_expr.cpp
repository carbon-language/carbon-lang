// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"

namespace Carbon::Check {

auto HandleIfExprIf(Context& context, Parse::IfExprIfId node_id) -> bool {
  // Alias node_id for if/then/else consistency.
  auto& if_node = node_id;

  auto [cond_node, cond_value_id] = context.node_stack().PopExprWithNodeId();

  // Convert the condition to `bool`, and branch on it.
  cond_value_id = ConvertToBoolValue(context, if_node, cond_value_id);
  context.node_stack().Push(cond_node, cond_value_id);
  auto then_block_id =
      context.AddDominatedBlockAndBranchIf(if_node, cond_value_id);
  auto else_block_id = context.AddDominatedBlockAndBranch(if_node);

  // Start emitting the `then` block.
  context.inst_block_stack().Pop();
  context.inst_block_stack().Push(then_block_id);
  context.AddCurrentCodeBlockToFunction(node_id);

  context.node_stack().Push(if_node, else_block_id);
  return true;
}

auto HandleIfExprThen(Context& context, Parse::IfExprThenId node_id) -> bool {
  auto then_value_id = context.node_stack().PopExpr();
  auto else_block_id = context.node_stack().Peek<Parse::NodeKind::IfExprIf>();

  // Convert the first operand to a value.
  then_value_id = ConvertToValueExpr(context, then_value_id);

  // Start emitting the `else` block.
  context.inst_block_stack().Push(else_block_id);
  context.AddCurrentCodeBlockToFunction(node_id);

  context.node_stack().Push(node_id, then_value_id);
  return true;
}

auto HandleIfExprElse(Context& context, Parse::IfExprElseId node_id) -> bool {
  // Alias node_id for if/then/else consistency.
  auto& else_node = node_id;

  auto else_value_id = context.node_stack().PopExpr();
  auto then_value_id = context.node_stack().Pop<Parse::NodeKind::IfExprThen>();
  auto [if_node, _] =
      context.node_stack().PopWithNodeId<Parse::NodeKind::IfExprIf>();
  auto cond_value_id = context.node_stack().PopExpr();

  // Convert the `else` value to the `then` value's type, and finish the `else`
  // block.
  // TODO: Find a common type, and convert both operands to it instead.
  auto result_type_id = context.insts().Get(then_value_id).type_id();
  else_value_id =
      ConvertToValueOfType(context, else_node, else_value_id, result_type_id);

  // Create a resumption block and branches to it.
  auto chosen_value_id = context.AddConvergenceBlockWithArgAndPush(
      if_node, {else_value_id, then_value_id});
  context.SetBlockArgResultBeforeConstantUse(chosen_value_id, cond_value_id,
                                             then_value_id, else_value_id);
  context.AddCurrentCodeBlockToFunction(node_id);

  // Push the result value.
  context.node_stack().Push(else_node, chosen_value_id);
  return true;
}

}  // namespace Carbon::Check
