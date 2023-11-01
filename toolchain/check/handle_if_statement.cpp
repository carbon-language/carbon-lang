// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

auto HandleIfConditionStart(Context& /*context*/, Parse::Lamp /*parse_lamp*/)
    -> bool {
  return true;
}

auto HandleIfCondition(Context& context, Parse::Lamp parse_lamp) -> bool {
  // Convert the condition to `bool`.
  auto cond_value_id = context.lamp_stack().PopExpression();
  cond_value_id = ConvertToBoolValue(context, parse_lamp, cond_value_id);

  // Create the then block and the else block, and branch to the right one. If
  // there is no `else`, the then block will terminate with a branch to the
  // else block, which will be reused as the resumption block.
  auto then_block_id =
      context.AddDominatedBlockAndBranchIf(parse_lamp, cond_value_id);
  auto else_block_id = context.AddDominatedBlockAndBranch(parse_lamp);

  // Start emitting the `then` block.
  context.inst_block_stack().Pop();
  context.inst_block_stack().Push(then_block_id);
  context.AddCurrentCodeBlockToFunction();

  context.lamp_stack().Push(parse_lamp, else_block_id);
  return true;
}

auto HandleIfStatementElse(Context& context, Parse::Lamp parse_lamp) -> bool {
  auto else_block_id = context.lamp_stack().Pop<Parse::LampKind::IfCondition>();

  // Switch to emitting the `else` block.
  context.inst_block_stack().Push(else_block_id);
  context.AddCurrentCodeBlockToFunction();

  context.lamp_stack().Push(parse_lamp);
  return true;
}

auto HandleIfStatement(Context& context, Parse::Lamp parse_lamp) -> bool {
  switch (auto kind = context.parse_tree().node_kind(
              context.lamp_stack().PeekParseLamp())) {
    case Parse::LampKind::IfCondition: {
      // Branch from then block to else block, and start emitting the else
      // block.
      auto else_block_id =
          context.lamp_stack().Pop<Parse::LampKind::IfCondition>();
      context.AddInst(SemIR::Branch{parse_lamp, else_block_id});
      context.inst_block_stack().Pop();
      context.inst_block_stack().Push(else_block_id);
      break;
    }

    case Parse::LampKind::IfStatementElse: {
      // Branch from the then and else blocks to a new resumption block.
      context.lamp_stack()
          .PopAndDiscardSoloParseLamp<Parse::LampKind::IfStatementElse>();
      context.AddConvergenceBlockAndPush(parse_lamp, /*num_blocks=*/2);
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
