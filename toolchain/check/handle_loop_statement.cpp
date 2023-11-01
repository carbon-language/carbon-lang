// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"

namespace Carbon::Check {

auto HandleBreakStatement(Context& /*context*/, Parse::Lamp /*parse_lamp*/)
    -> bool {
  return true;
}

auto HandleBreakStatementStart(Context& context, Parse::Lamp parse_lamp)
    -> bool {
  auto& stack = context.break_continue_stack();
  if (stack.empty()) {
    CARBON_DIAGNOSTIC(BreakOutsideLoop, Error,
                      "`break` can only be used in a loop.");
    context.emitter().Emit(parse_lamp, BreakOutsideLoop);
  } else {
    context.AddInst(SemIR::Branch{parse_lamp, stack.back().break_target});
  }

  context.inst_block_stack().Pop();
  context.inst_block_stack().PushUnreachable();
  return true;
}

auto HandleContinueStatement(Context& /*context*/, Parse::Lamp /*parse_lamp*/)
    -> bool {
  return true;
}

auto HandleContinueStatementStart(Context& context, Parse::Lamp parse_lamp)
    -> bool {
  auto& stack = context.break_continue_stack();
  if (stack.empty()) {
    CARBON_DIAGNOSTIC(ContinueOutsideLoop, Error,
                      "`continue` can only be used in a loop.");
    context.emitter().Emit(parse_lamp, ContinueOutsideLoop);
  } else {
    context.AddInst(SemIR::Branch{parse_lamp, stack.back().continue_target});
  }

  context.inst_block_stack().Pop();
  context.inst_block_stack().PushUnreachable();
  return true;
}

auto HandleForHeader(Context& context, Parse::Lamp parse_lamp) -> bool {
  return context.TODO(parse_lamp, "HandleForHeader");
}

auto HandleForHeaderStart(Context& context, Parse::Lamp parse_lamp) -> bool {
  return context.TODO(parse_lamp, "HandleForHeaderStart");
}

auto HandleForIn(Context& context, Parse::Lamp parse_lamp) -> bool {
  return context.TODO(parse_lamp, "HandleForIn");
}

auto HandleForStatement(Context& context, Parse::Lamp parse_lamp) -> bool {
  return context.TODO(parse_lamp, "HandleForStatement");
}

auto HandleWhileConditionStart(Context& context, Parse::Lamp parse_lamp)
    -> bool {
  // Branch to the loop header block. Note that we create a new block here even
  // if the current block is empty; this ensures that the loop always has a
  // preheader block.
  auto loop_header_id = context.AddDominatedBlockAndBranch(parse_lamp);
  context.inst_block_stack().Pop();

  // Start emitting the loop header block.
  context.inst_block_stack().Push(loop_header_id);
  context.AddCurrentCodeBlockToFunction();

  context.lamp_stack().Push(parse_lamp, loop_header_id);
  return true;
}

auto HandleWhileCondition(Context& context, Parse::Lamp parse_lamp) -> bool {
  auto cond_value_id = context.lamp_stack().PopExpression();
  auto loop_header_id =
      context.lamp_stack().Peek<Parse::LampKind::WhileConditionStart>();
  cond_value_id = ConvertToBoolValue(context, parse_lamp, cond_value_id);

  // Branch to either the loop body or the loop exit block.
  auto loop_body_id =
      context.AddDominatedBlockAndBranchIf(parse_lamp, cond_value_id);
  auto loop_exit_id = context.AddDominatedBlockAndBranch(parse_lamp);
  context.inst_block_stack().Pop();

  // Start emitting the loop body.
  context.inst_block_stack().Push(loop_body_id);
  context.AddCurrentCodeBlockToFunction();
  context.break_continue_stack().push_back(
      {.break_target = loop_exit_id, .continue_target = loop_header_id});

  context.lamp_stack().Push(parse_lamp, loop_exit_id);
  return true;
}

auto HandleWhileStatement(Context& context, Parse::Lamp parse_lamp) -> bool {
  auto loop_exit_id =
      context.lamp_stack().Pop<Parse::LampKind::WhileCondition>();
  auto loop_header_id =
      context.lamp_stack().Pop<Parse::LampKind::WhileConditionStart>();
  context.break_continue_stack().pop_back();

  // Add the loop backedge.
  context.AddInst(SemIR::Branch{parse_lamp, loop_header_id});
  context.inst_block_stack().Pop();

  // Start emitting the loop exit block.
  context.inst_block_stack().Push(loop_exit_id);
  context.AddCurrentCodeBlockToFunction();
  return true;
}

}  // namespace Carbon::Check
