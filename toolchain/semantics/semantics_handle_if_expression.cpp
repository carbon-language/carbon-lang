// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"

namespace Carbon {

auto SemanticsHandleIfExpressionIf(SemanticsContext& context,
                                   ParseTree::Node if_node) -> bool {
  auto cond_value_id = context.node_stack().Pop<SemanticsNodeId>();

  context.node_stack().Push(if_node);

  // Convert the condition to `bool`.
  cond_value_id = context.ImplicitAsBool(if_node, cond_value_id);

  // Stop emitting the current block. We'll add some branch instructions to it
  // later, but we don't want it on the stack any more.
  auto if_block_id = context.node_block_stack().PeekForAdd();
  context.node_block_stack().Pop();

  // Create the resumption block, `else` block, and `then` block, and branches
  // to them.
  context.node_block_stack().Push();
  auto else_block_id = context.node_block_stack().PushForAdd();
  auto then_block_id = context.node_block_stack().PushForAdd();
  context.AddNodeToBlock(
      if_block_id,
      SemanticsNode::BranchIf::Make(if_node, then_block_id, cond_value_id));
  context.AddNodeToBlock(if_block_id,
                         SemanticsNode::Branch::Make(if_node, else_block_id));
  return true;
}

auto SemanticsHandleIfExpressionThen(SemanticsContext& context,
                                     ParseTree::Node then_node) -> bool {
  context.node_stack().Push(then_node, context.node_block_stack().Pop());
  return true;
}

auto SemanticsHandleIfExpressionElse(SemanticsContext& context,
                                     ParseTree::Node else_node) -> bool {
  auto else_value_id = context.node_stack().Pop<SemanticsNodeId>();
  auto [then_node, then_end_block_id] =
      context.node_stack().PopWithParseNode<SemanticsNodeBlockId>(
          ParseNodeKind::IfExpressionThen);
  auto then_value_id = context.node_stack().Pop<SemanticsNodeId>();
  auto if_node =
      context.node_stack().PopForSoloParseNode(ParseNodeKind::IfExpressionIf);

  // Convert the `else` value to the `then` value's type, and finish the `else`
  // block.
  // TODO: Find a common type, and convert both operands to it instead.
  auto result_type_id = context.semantics_ir().GetNode(then_value_id).type_id();
  else_value_id =
      context.ImplicitAsRequired(else_node, else_value_id, result_type_id);
  auto else_end_block_id = context.node_block_stack().Pop();

  // Create branches to the resumption block.
  auto resume_block_id = context.node_block_stack().PeekForAdd();
  context.AddNodeToBlock(then_end_block_id,
                         SemanticsNode::BranchWithArg::Make(
                             then_node, resume_block_id, then_value_id));
  context.AddNodeToBlock(else_end_block_id,
                         SemanticsNode::BranchWithArg::Make(
                             else_node, resume_block_id, else_value_id));

  // Obtain the value in the resumption block and push it.
  context.AddNodeAndPush(
      if_node, SemanticsNode::BlockArg::Make(if_node, result_type_id));
  return true;
}

}  // namespace Carbon
