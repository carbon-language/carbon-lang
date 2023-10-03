// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"

namespace Carbon::Check {

auto HandleBreakStatement(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleBreakStatement");
}

auto HandleBreakStatementStart(Context& context, Parse::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleBreakStatementStart");
}

auto HandleContinueStatement(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleContinueStatement");
}

auto HandleContinueStatementStart(Context& context, Parse::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleContinueStatementStart");
}

auto HandleForHeader(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleForHeader");
}

auto HandleForHeaderStart(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleForHeaderStart");
}

auto HandleForIn(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleForIn");
}

auto HandleForStatement(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleForStatement");
}

auto HandleWhileConditionStart(Context& context, Parse::Node parse_node)
    -> bool {
  // Branch to the loop entry block.
  auto loop_entry_id = context.AddDominatedBlockAndBranch(parse_node);
  context.node_block_stack().Pop();

  // Start emitting the loop entry block.
  context.node_block_stack().Push(loop_entry_id);
  context.AddCurrentCodeBlockToFunction();

  context.node_stack().Push(parse_node, loop_entry_id);
  return true;
}

auto HandleWhileCondition(Context& context, Parse::Node parse_node) -> bool {
  auto cond_value_id = context.node_stack().PopExpression();
  cond_value_id = ConvertToBoolValue(context, parse_node, cond_value_id);

  // Branch to either the loop body or the loop exit block.
  auto loop_body_id =
      context.AddDominatedBlockAndBranchIf(parse_node, cond_value_id);
  auto loop_exit_id = context.AddDominatedBlockAndBranch(parse_node);
  context.node_block_stack().Pop();

  // Start emitting the loop body.
  context.node_block_stack().Push(loop_body_id);
  context.AddCurrentCodeBlockToFunction();

  context.node_stack().Push(parse_node, loop_exit_id);
  return true;
}

auto HandleWhileStatement(Context& context, Parse::Node parse_node) -> bool {
  auto loop_exit_id =
      context.node_stack().Pop<Parse::NodeKind::WhileCondition>();
  auto loop_entry_id =
      context.node_stack().Pop<Parse::NodeKind::WhileConditionStart>();

  // Add the loop backedge.
  context.AddNode(SemIR::Node::Branch::Make(parse_node, loop_entry_id));
  context.node_block_stack().Pop();

  // Start emitting the loop exit block.
  context.node_block_stack().Push(loop_exit_id);
  context.AddCurrentCodeBlockToFunction();
  return true;
}

}  // namespace Carbon::Check
