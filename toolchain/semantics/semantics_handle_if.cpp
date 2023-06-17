// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

auto SemanticsHandleIfConditionStart(SemanticsContext& /*context*/,
                                     ParseTree::Node /*parse_node*/) -> bool {
  return true;
}

auto SemanticsHandleIfCondition(SemanticsContext& context,
                                ParseTree::Node parse_node) -> bool {
  // Convert the condition to `bool`.
  auto cond_value_id = context.node_stack().Pop<SemanticsNodeId>();
  cond_value_id = context.ImplicitAsBool(parse_node, cond_value_id);

  // Create the else block and the then block, and branch to the right one. If
  // there is no `else`, the then block will terminate with a branch to the
  // else block, which will be reused as the resumption block.
  auto if_block_id = context.node_block_stack().PopForAdd();
  auto else_block_id = context.node_block_stack().PushForAdd();
  auto then_block_id = context.node_block_stack().PushForAdd();

  // Branch to the appropriate block.
  context.AddNodeToBlock(
      if_block_id,
      SemanticsNode::BranchIf::Make(parse_node, then_block_id, cond_value_id));
  context.AddNodeToBlock(
      if_block_id, SemanticsNode::Branch::Make(parse_node, else_block_id));

  context.node_stack().Push(parse_node);
  return true;
}

auto SemanticsHandleIfStatementElse(SemanticsContext& context,
                                    ParseTree::Node parse_node) -> bool {
  context.node_stack().PopAndDiscardSoloParseNode(ParseNodeKind::IfCondition);

  // Switch to emitting the else block.
  auto then_block_id = context.node_block_stack().PopForAdd();
  context.node_stack().Push(parse_node, then_block_id);
  return true;
}

auto SemanticsHandleIfStatement(SemanticsContext& context,
                                ParseTree::Node parse_node) -> bool {
  // Either the then or else block, depending on whether there's an `else` node
  // on the top of the node stack.
  auto sub_block_id = context.node_block_stack().PopForAdd();

  switch (auto kind = context.parse_tree().node_kind(
              context.node_stack().PeekParseNode())) {
    case ParseNodeKind::IfCondition: {
      // Branch from then block to else block.
      context.node_stack().PopAndDiscardSoloParseNode(
          ParseNodeKind::IfCondition);
      context.AddNodeToBlock(
          sub_block_id,
          SemanticsNode::Branch::Make(parse_node,
                                      context.node_block_stack().PeekForAdd()));
      break;
    }

    case ParseNodeKind::IfStatementElse: {
      // Branch from the then and else blocks to a new resumption block.
      auto then_block_id = context.node_stack().Pop<SemanticsNodeBlockId>(
          ParseNodeKind::IfStatementElse);
      auto resume_block_id = context.node_block_stack().PushForAdd();
      context.AddNodeToBlock(then_block_id, SemanticsNode::Branch::Make(
                                                parse_node, resume_block_id));
      context.AddNodeToBlock(sub_block_id, SemanticsNode::Branch::Make(
                                               parse_node, resume_block_id));
      break;
    }

    default: {
      CARBON_FATAL() << "Unexpected parse node at start of `if`: " << kind;
    }
  }

  return true;
}

}  // namespace Carbon
