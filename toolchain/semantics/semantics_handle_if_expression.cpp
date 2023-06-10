// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"

namespace Carbon {

auto SemanticsHandleIfExpressionThen(SemanticsContext& context,
                                     ParseTree::Node then_node) -> bool {
  // TODO: Find the `if` node.
  ParseTree::Node if_node = then_node;

  // Convert the condition to `bool`.
  auto cond_value = context.node_stack().Pop<SemanticsNodeId>();
  cond_value = context.ImplicitAsRequired(
      if_node, cond_value,
      context.CanonicalizeType(SemanticsNodeId::BuiltinBoolType));
  context.node_stack().Push(if_node, cond_value);

  // Stop emitting the current block. We'll add some branch instructions to it
  // later, but we don't want it on the stack any more.
  auto if_block = context.node_block_stack().PeekForAdd();
  context.node_stack().Push(if_node, if_block);
  context.node_block_stack().Pop();

  // Create the `then` block.
  context.node_block_stack().Push();
  auto then_block = context.node_block_stack().PeekForAdd();
  context.node_stack().Push(then_node, then_block);
  return true;
}

auto SemanticsHandleIfExpressionElse(SemanticsContext& context,
                                     ParseTree::Node else_node) -> bool {
  context.node_stack().Push(else_node, context.node_block_stack().Pop());

  // Create the `else` block.
  context.node_block_stack().Push();
  auto else_block = context.node_block_stack().PeekForAdd();
  context.node_stack().Push(else_node, else_block);
  return true;
}

auto SemanticsHandleIfExpression(SemanticsContext& context,
                                 ParseTree::Node parse_node) -> bool {
  auto else_value = context.node_stack().Pop<SemanticsNodeId>();
  auto [else_node, else_block] =
      context.node_stack().PopWithParseNode<SemanticsNodeBlockId>();
  auto then_end_block = context.node_stack().Pop<SemanticsNodeBlockId>();
  auto then_value = context.node_stack().Pop<SemanticsNodeId>();
  auto [then_node, then_block] =
      context.node_stack().PopWithParseNode<SemanticsNodeBlockId>();
  auto if_block = context.node_stack().Pop<SemanticsNodeBlockId>();
  auto cond_value = context.node_stack().Pop<SemanticsNodeId>();

  // Convert the `else` value to the `then` value's type, and finish the `else`
  // block.
  // TODO: Find a common type, and convert both operands to it instead.
  auto result_type = context.semantics_ir().GetNode(then_value).type_id();
  else_value = context.ImplicitAsRequired(parse_node, else_value, result_type);
  auto else_end_block = context.node_block_stack().Pop();

  // Create a resumption block.
  context.node_block_stack().Push();
  auto resume_block = context.node_block_stack().PeekForAdd();

  // Add branches from the end of the `if` block to the `then` and `else`
  // blocks.
  context.semantics_ir().AddNode(
      if_block,
      SemanticsNode::BranchIf::Make(parse_node, then_block, cond_value));
  context.semantics_ir().AddNode(
      if_block, SemanticsNode::Branch::Make(parse_node, else_block));

  // Add branches from the end of the `then` and `else` blocks to the
  // resumption block.
  context.semantics_ir().AddNode(
      then_end_block,
      SemanticsNode::BranchWithArg::Make(then_node, resume_block, then_value));
  context.semantics_ir().AddNode(
      else_end_block,
      SemanticsNode::BranchWithArg::Make(else_node, resume_block, else_value));

  // Obtain the value in the resumption block and push it.
  context.AddNodeAndPush(
      parse_node, SemanticsNode::BlockArg::Make(parse_node, result_type));
  return true;
}

}  // namespace Carbon
