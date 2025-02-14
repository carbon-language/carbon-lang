// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/control_flow.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"

namespace Carbon::Check {

// `while`
// -------

auto HandleParseNode(Context& context, Parse::WhileConditionStartId node_id)
    -> bool {
  // Branch to the loop header block. Note that we create a new block here even
  // if the current block is empty; this ensures that the loop always has a
  // preheader block.
  auto loop_header_id = AddDominatedBlockAndBranch(context, node_id);
  context.inst_block_stack().Pop();

  // Start emitting the loop header block.
  context.inst_block_stack().Push(loop_header_id);
  context.region_stack().AddToRegion(loop_header_id, node_id);

  context.node_stack().Push(node_id, loop_header_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::WhileConditionId node_id)
    -> bool {
  auto cond_value_id = context.node_stack().PopExpr();
  auto loop_header_id =
      context.node_stack().Peek<Parse::NodeKind::WhileConditionStart>();
  cond_value_id = ConvertToBoolValue(context, node_id, cond_value_id);

  // Branch to either the loop body or the loop exit block.
  auto loop_body_id =
      AddDominatedBlockAndBranchIf(context, node_id, cond_value_id);
  auto loop_exit_id = AddDominatedBlockAndBranch(context, node_id);
  context.inst_block_stack().Pop();

  // Start emitting the loop body.
  context.inst_block_stack().Push(loop_body_id);
  context.region_stack().AddToRegion(loop_body_id, node_id);
  context.break_continue_stack().push_back(
      {.break_target = loop_exit_id, .continue_target = loop_header_id});

  context.node_stack().Push(node_id, loop_exit_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::WhileStatementId node_id)
    -> bool {
  auto loop_exit_id =
      context.node_stack().Pop<Parse::NodeKind::WhileCondition>();
  auto loop_header_id =
      context.node_stack().Pop<Parse::NodeKind::WhileConditionStart>();
  context.break_continue_stack().pop_back();

  // Add the loop backedge.
  AddInst<SemIR::Branch>(context, node_id, {.target_id = loop_header_id});
  context.inst_block_stack().Pop();

  // Start emitting the loop exit block.
  context.inst_block_stack().Push(loop_exit_id);
  context.region_stack().AddToRegion(loop_exit_id, node_id);
  return true;
}

// `for`
// -----

auto HandleParseNode(Context& context, Parse::ForHeaderStartId node_id)
    -> bool {
  return context.TODO(node_id, "HandleForHeaderStart");
}

auto HandleParseNode(Context& context, Parse::ForInId node_id) -> bool {
  context.decl_introducer_state_stack().Pop<Lex::TokenKind::Var>();
  return context.TODO(node_id, "HandleForIn");
}

auto HandleParseNode(Context& context, Parse::ForHeaderId node_id) -> bool {
  return context.TODO(node_id, "HandleForHeader");
}

auto HandleParseNode(Context& context, Parse::ForStatementId node_id) -> bool {
  return context.TODO(node_id, "HandleForStatement");
}

// `break`
// -------

auto HandleParseNode(Context& context, Parse::BreakStatementStartId node_id)
    -> bool {
  auto& stack = context.break_continue_stack();
  if (stack.empty()) {
    CARBON_DIAGNOSTIC(BreakOutsideLoop, Error,
                      "`break` can only be used in a loop");
    context.emitter().Emit(node_id, BreakOutsideLoop);
  } else {
    AddInst<SemIR::Branch>(context, node_id,
                           {.target_id = stack.back().break_target});
  }

  context.inst_block_stack().Pop();
  context.inst_block_stack().PushUnreachable();
  return true;
}

auto HandleParseNode(Context& /*context*/, Parse::BreakStatementId /*node_id*/)
    -> bool {
  return true;
}

// `continue`
// ----------

auto HandleParseNode(Context& context, Parse::ContinueStatementStartId node_id)
    -> bool {
  auto& stack = context.break_continue_stack();
  if (stack.empty()) {
    CARBON_DIAGNOSTIC(ContinueOutsideLoop, Error,
                      "`continue` can only be used in a loop");
    context.emitter().Emit(node_id, ContinueOutsideLoop);
  } else {
    AddInst<SemIR::Branch>(context, node_id,
                           {.target_id = stack.back().continue_target});
  }

  context.inst_block_stack().Pop();
  context.inst_block_stack().PushUnreachable();
  return true;
}

auto HandleParseNode(Context& /*context*/,
                     Parse::ContinueStatementId /*node_id*/) -> bool {
  return true;
}

}  // namespace Carbon::Check
