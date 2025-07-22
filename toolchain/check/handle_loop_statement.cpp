// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/call.h"
#include "toolchain/check/context.h"
#include "toolchain/check/control_flow.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/full_pattern_stack.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/member_access.h"
#include "toolchain/check/operator.h"
#include "toolchain/check/pattern.h"
#include "toolchain/check/pattern_match.h"
#include "toolchain/check/type.h"
#include "toolchain/sem_ir/absolute_node_id.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Starts emitting the loop header for a `while`-like looping construct. Returns
// the loop header block ID.
static auto StartLoopHeader(Context& context, Parse::NodeId node_id)
    -> SemIR::InstBlockId {
  // Branch to the loop header block. Note that we create a new block here even
  // if the current block is empty; this ensures that the loop always has a
  // preheader block.
  auto loop_header_id = AddDominatedBlockAndBranch(context, node_id);
  context.inst_block_stack().Pop();

  // Start emitting the loop header block.
  context.inst_block_stack().Push(loop_header_id);
  context.region_stack().AddToRegion(loop_header_id, node_id);

  return loop_header_id;
}

// Starts emitting the loop body for a `while`-like looping construct. Converts
// `cond_value_id` to bool and branches to the loop body if it is `true` and to
// the loop exit if it is `false`.
static auto BranchAndStartLoopBody(Context& context, Parse::NodeId node_id,
                                   SemIR::InstBlockId loop_header_id,
                                   SemIR::InstId cond_value_id) -> void {
  cond_value_id = ConvertToBoolValue(context, node_id, cond_value_id);

  // Branch to either the loop body or the loop exit block.
  auto loop_body_id =
      AddDominatedBlockAndBranchIf(context, node_id, cond_value_id);
  auto loop_exit_id = AddDominatedBlockAndBranch(context, node_id);
  context.inst_block_stack().Pop();

  // Start emitting the loop body.
  context.inst_block_stack().Push(loop_body_id);
  context.region_stack().AddToRegion(loop_body_id, node_id);

  // Allow `break` and `continue` in this scope.
  context.break_continue_stack().push_back(
      {.break_target = loop_exit_id, .continue_target = loop_header_id});
}

// Finishes emitting the body for a `while`-like loop. Adds a back-edge to the
// loop header, and starts emitting in the loop exit block.
static auto FinishLoopBody(Context& context, Parse::NodeId node_id) -> void {
  auto blocks = context.break_continue_stack().pop_back_val();

  // Add the loop backedge.
  AddInst<SemIR::Branch>(context, node_id,
                         {.target_id = blocks.continue_target});
  context.inst_block_stack().Pop();

  // Start emitting the loop exit block.
  context.inst_block_stack().Push(blocks.break_target);
  context.region_stack().AddToRegion(blocks.break_target, node_id);
}

// `while`
// -------

auto HandleParseNode(Context& context, Parse::WhileConditionStartId node_id)
    -> bool {
  context.node_stack().Push(node_id, StartLoopHeader(context, node_id));
  return true;
}

auto HandleParseNode(Context& context, Parse::WhileConditionId node_id)
    -> bool {
  auto cond_value_id = context.node_stack().PopExpr();
  auto loop_header_id =
      context.node_stack().Pop<Parse::NodeKind::WhileConditionStart>();

  // Branch to either the loop body or the loop exit block, and start emitting
  // the loop body.
  BranchAndStartLoopBody(context, node_id, loop_header_id, cond_value_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::WhileStatementId node_id)
    -> bool {
  FinishLoopBody(context, node_id);
  return true;
}

// `for`
// -----

auto HandleParseNode(Context& context, Parse::ForHeaderStartId node_id)
    -> bool {
  // Create a nested scope to hold the cursor variable. This is also the lexical
  // scope that names in the pattern are added to, although they get rebound on
  // each loop iteration.
  context.scope_stack().PushForSameRegion();

  // Begin an implicit let declaration context for the pattern.
  context.decl_introducer_state_stack().Push<Lex::TokenKind::Let>();
  context.pattern_block_stack().Push();
  context.full_pattern_stack().PushFullPattern(
      FullPatternStack::Kind::NameBindingDecl);
  BeginSubpattern(context);

  context.node_stack().Push(node_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::ForInId node_id) -> bool {
  auto pattern_block_id = context.pattern_block_stack().Pop();
  AddInst<SemIR::NameBindingDecl>(context, node_id,
                                  {.pattern_block_id = pattern_block_id});
  context.decl_introducer_state_stack().Pop<Lex::TokenKind::Let>();
  context.full_pattern_stack().StartPatternInitializer();
  context.node_stack().Push(node_id, pattern_block_id);
  return true;
}

// For a value or reference of type `Optional(T)`, call the given accessor.
static auto CallOptionalAccessor(Context& context, Parse::NodeId node_id,
                                 SemIR::InstId optional_id,
                                 llvm::StringLiteral accessor_name)
    -> SemIR::InstId {
  auto accessor_name_id =
      SemIR::NameId::ForIdentifier(context.identifiers().Add(accessor_name));
  auto accessor_id =
      PerformMemberAccess(context, node_id, optional_id, accessor_name_id);
  return PerformCall(context, node_id, accessor_id, {});
}

auto HandleParseNode(Context& context, Parse::ForHeaderId node_id) -> bool {
  auto range_id = context.node_stack().PopExpr();
  auto pattern_block_id = context.node_stack().Pop<Parse::NodeKind::ForIn>();
  auto pattern_id = context.node_stack().PopPattern();
  auto start_node_id =
      context.node_stack().PopForSoloNodeId<Parse::NodeKind::ForHeaderStart>();

  // Convert the range expression to a value or reference so that we can use it
  // multiple times.
  // TODO: If this produces a temporary, its lifetime should presumably be
  // extended to cover the loop body.
  range_id = ConvertToValueOrRefExpr(context, range_id);

  // Create the cursor variable.
  // TODO: Produce a custom diagnostic if the range operand can't be used as a
  // range.
  auto cursor_id = BuildUnaryOperator(
      context, node_id, {.interface_name = "Iterate", .op_name = "NewCursor"},
      range_id);
  auto cursor_type_id = context.insts().Get(cursor_id).type_id();
  auto cursor_var_id = AddInstWithCleanup<SemIR::VarStorage>(
      context, node_id,
      {.type_id = cursor_type_id, .pattern_id = SemIR::AbsoluteInstId::None});
  auto init_id = Initialize(context, node_id, cursor_var_id, cursor_id);
  AddInst<SemIR::Assign>(context, node_id,
                         {.lhs_id = cursor_var_id, .rhs_id = init_id});

  // Start emitting the loop header block.
  auto loop_header_id = StartLoopHeader(context, start_node_id);

  // Call `<range>.(Iterate.Next)(&cursor)`.
  auto cursor_type_inst_id = context.types().GetInstId(cursor_type_id);
  auto cursor_addr_id = AddInst<SemIR::AddrOf>(
      context, node_id,
      {.type_id = GetPointerType(context, cursor_type_inst_id),
       .lvalue_id = cursor_var_id});
  auto element_id = BuildBinaryOperator(
      context, node_id, {.interface_name = "Iterate", .op_name = "Next"},
      range_id, cursor_addr_id);
  // We need to convert away from an initializing expression in order to call
  // `HasValue` and then separately pattern-match against the element.
  // TODO: Instead, form a `.Some(pattern_id)` pattern and pattern-match against
  // that.
  element_id = ConvertToValueOrRefExpr(context, element_id);

  // Branch to the loop body if the optional element has a value.
  auto cond_value_id =
      CallOptionalAccessor(context, node_id, element_id, "HasValue");
  BranchAndStartLoopBody(context, node_id, loop_header_id, cond_value_id);

  // The loop pattern's initializer is now complete, and any bindings in it
  // should be in scope.
  context.full_pattern_stack().EndPatternInitializer();
  context.full_pattern_stack().PopFullPattern();

  // Create storage for var patterns now.
  AddPatternVarStorage(context, pattern_block_id, /*is_returned_var=*/false);

  // Initialize the pattern from `<element>.Get()`.
  auto element_value_id =
      CallOptionalAccessor(context, node_id, element_id, "Get");
  LocalPatternMatch(context, pattern_id, element_value_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::ForStatementId node_id) -> bool {
  FinishLoopBody(context, node_id);
  return true;
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
