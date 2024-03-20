// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"

namespace Carbon::Check {

auto HandleIfConditionStart(Context& /*context*/,
                            Parse::IfConditionStartId /*node_id*/) -> bool {
  return true;
}

auto HandleIfCondition(Context& context, Parse::IfConditionId node_id) -> bool {
  // Convert the condition to `bool`.
  auto cond_value_id = context.node_stack().PopExpr();
  cond_value_id = ConvertToBoolValue(context, node_id, cond_value_id);

  // Create the then block and the else block, and branch to the right one. If
  // there is no `else`, the then block will terminate with a branch to the
  // else block, which will be reused as the resumption block.
  auto then_block_id =
      context.AddDominatedBlockAndBranchIf(node_id, cond_value_id);
  auto else_block_id = context.AddDominatedBlockAndBranch(node_id);

  // Start emitting the `then` block.
  context.inst_block_stack().Pop();
  context.inst_block_stack().Push(then_block_id);
  context.AddCurrentCodeBlockToFunction();

  context.node_stack().Push(node_id, else_block_id);
  return true;
}

auto HandleIfStatementElse(Context& context, Parse::IfStatementElseId node_id)
    -> bool {
  auto else_block_id = context.node_stack().Pop<Parse::NodeKind::IfCondition>();

  // Switch to emitting the `else` block.
  context.inst_block_stack().Push(else_block_id);
  context.AddCurrentCodeBlockToFunction();

  context.node_stack().Push(node_id);
  return true;
}

auto HandleIfStatement(Context& context, Parse::IfStatementId node_id) -> bool {
  switch (auto kind = context.node_stack().PeekNodeKind()) {
    case Parse::NodeKind::IfCondition: {
      // Branch from then block to else block, and start emitting the else
      // block.
      auto else_block_id =
          context.node_stack().Pop<Parse::NodeKind::IfCondition>();
      context.AddInst({node_id, SemIR::Branch{else_block_id}});
      context.inst_block_stack().Pop();
      context.inst_block_stack().Push(else_block_id);
      break;
    }

    case Parse::NodeKind::IfStatementElse: {
      // Branch from the then and else blocks to a new resumption block.
      context.node_stack()
          .PopAndDiscardSoloNodeId<Parse::NodeKind::IfStatementElse>();
      context.AddConvergenceBlockAndPush(node_id, /*num_blocks=*/2);
      break;
    }

    default: {
      CARBON_FATAL() << "Unexpected parse node at start of `if`: " << kind;
    }
  }

  context.AddCurrentCodeBlockToFunction();
  return true;
}

}  // namespace Carbon::Check
