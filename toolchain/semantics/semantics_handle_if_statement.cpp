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
  auto cond_value_id = context.node_stack().PopExpression();
  cond_value_id = context.ImplicitAsBool(parse_node, cond_value_id);

  // Create the then block and the else block, and branch to the right one. If
  // there is no `else`, the then block will terminate with a branch to the
  // else block, which will be reused as the resumption block.
  auto then_block_id =
      context.AddDominatedBlockAndBranchIf(parse_node, cond_value_id);
  auto else_block_id = context.AddDominatedBlockAndBranch(parse_node);

  // Push the else and then blocks, and start emitting code in the then block.
  context.node_block_stack().Pop();
  context.node_block_stack().Push(else_block_id);
  context.node_block_stack().Push(then_block_id);
  context.AddCurrentCodeBlockToFunction();

  context.node_stack().Push(parse_node);
  return true;
}

auto SemanticsHandleIfStatementElse(SemanticsContext& context,
                                    ParseTree::Node parse_node) -> bool {
  context.node_stack().PopAndDiscardSoloParseNode<ParseNodeKind::IfCondition>();

  // Switch to emitting the else block.
  auto then_block_id = context.node_block_stack().PopForAdd();
  context.node_stack().Push(parse_node, then_block_id);
  context.AddCurrentCodeBlockToFunction();
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
      context.node_stack()
          .PopAndDiscardSoloParseNode<ParseNodeKind::IfCondition>();
      context.AddNodeToBlock(
          sub_block_id,
          SemanticsNode::Branch::Make(parse_node,
                                      context.node_block_stack().PeekForAdd()));
      break;
    }

    case ParseNodeKind::IfStatementElse: {
      // Branch from the then and else blocks to a new resumption block.
      SemanticsNodeBlockId then_block_id =
          context.node_stack().Pop<ParseNodeKind::IfStatementElse>();
      context.AddConvergenceBlockAndPush(parse_node,
                                         {then_block_id, sub_block_id});
      break;
    }

    default: {
      CARBON_FATAL() << "Unexpected parse node at start of `if`: " << kind;
    }
  }

  context.AddCurrentCodeBlockToFunction();
  return true;
}

}  // namespace Carbon
