// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"

namespace Carbon::Check {

// `while`
// -------

auto HandleWhileConditionStart(Context& context,
                               Parse::WhileConditionStartId parse_node)
    -> bool {
  // Branch to the loop header block. Note that we create a new block here even
  // if the current block is empty; this ensures that the loop always has a
  // preheader block.
  auto loop_header_id = context.AddDominatedBlockAndBranch(parse_node);
  context.inst_block_stack().Pop();

  // Start emitting the loop header block.
  context.inst_block_stack().Push(loop_header_id);
  context.AddCurrentCodeBlockToFunction();

  context.node_stack().Push(parse_node, loop_header_id);
  return true;
}

auto HandleWhileCondition(Context& context, Parse::WhileConditionId parse_node)
    -> bool {
  auto cond_value_id = context.node_stack().PopExpr();
  auto loop_header_id =
      context.node_stack().Peek<Parse::NodeKind::WhileConditionStart>();
  cond_value_id = ConvertToBoolValue(context, parse_node, cond_value_id);

  // Branch to either the loop body or the loop exit block.
  auto loop_body_id =
      context.AddDominatedBlockAndBranchIf(parse_node, cond_value_id);
  auto loop_exit_id = context.AddDominatedBlockAndBranch(parse_node);
  context.inst_block_stack().Pop();

  // Start emitting the loop body.
  context.inst_block_stack().Push(loop_body_id);
  context.AddCurrentCodeBlockToFunction();
  context.break_continue_stack().push_back(
      {.break_target = loop_exit_id, .continue_target = loop_header_id});

  context.node_stack().Push(parse_node, loop_exit_id);
  return true;
}

auto HandleWhileStatement(Context& context, Parse::WhileStatementId parse_node)
    -> bool {
  auto loop_exit_id =
      context.node_stack().Pop<Parse::NodeKind::WhileCondition>();
  auto loop_header_id =
      context.node_stack().Pop<Parse::NodeKind::WhileConditionStart>();
  context.break_continue_stack().pop_back();

  // Add the loop backedge.
  context.AddInst({parse_node, SemIR::Branch{loop_header_id}});
  context.inst_block_stack().Pop();

  // Start emitting the loop exit block.
  context.inst_block_stack().Push(loop_exit_id);
  context.AddCurrentCodeBlockToFunction();
  return true;
}

// `for`
// -----

auto HandleForHeaderStart(Context& context, Parse::ForHeaderStartId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleForHeaderStart");
}

auto HandleForIn(Context& context, Parse::ForInId parse_node) -> bool {
  context.decl_state_stack().Pop(DeclState::Var);
  return context.TODO(parse_node, "HandleForIn");
}

auto HandleForHeader(Context& context, Parse::ForHeaderId parse_node) -> bool {
  return context.TODO(parse_node, "HandleForHeader");
}

auto HandleForStatement(Context& context, Parse::ForStatementId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleForStatement");
}

// `break`
// -------

auto HandleBreakStatementStart(Context& context,
                               Parse::BreakStatementStartId parse_node)
    -> bool {
  auto& stack = context.break_continue_stack();
  if (stack.empty()) {
    CARBON_DIAGNOSTIC(BreakOutsideLoop, Error,
                      "`break` can only be used in a loop.");
    context.emitter().Emit(parse_node, BreakOutsideLoop);
  } else {
    context.AddInst({parse_node, SemIR::Branch{stack.back().break_target}});
  }

  context.inst_block_stack().Pop();
  context.inst_block_stack().PushUnreachable();
  return true;
}

auto HandleBreakStatement(Context& /*context*/,
                          Parse::BreakStatementId /*parse_node*/) -> bool {
  return true;
}

// `continue`
// ----------

auto HandleContinueStatementStart(Context& context,
                                  Parse::ContinueStatementStartId parse_node)
    -> bool {
  auto& stack = context.break_continue_stack();
  if (stack.empty()) {
    CARBON_DIAGNOSTIC(ContinueOutsideLoop, Error,
                      "`continue` can only be used in a loop.");
    context.emitter().Emit(parse_node, ContinueOutsideLoop);
  } else {
    context.AddInst({parse_node, SemIR::Branch{stack.back().continue_target}});
  }

  context.inst_block_stack().Pop();
  context.inst_block_stack().PushUnreachable();
  return true;
}

auto HandleContinueStatement(Context& /*context*/,
                             Parse::ContinueStatementId /*parse_node*/)
    -> bool {
  return true;
}

}  // namespace Carbon::Check
