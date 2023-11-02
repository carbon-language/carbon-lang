// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

auto HandleBraceExpression(Context& context) -> void {
  auto state = context.PopState();

  state.state = State::BraceExpressionFinishAsUnknown;
  context.PushState(state);

  CARBON_CHECK(context.ConsumeAndAddLeafNodeIf(
      Lex::TokenKind::OpenCurlyBrace,
      NodeKind::StructLiteralOrStructTypeLiteralStart));
  if (!context.PositionIs(Lex::TokenKind::CloseCurlyBrace)) {
    context.PushState(State::BraceExpressionParameterAsUnknown);
  }
}

// Prints a diagnostic for brace expression syntax errors.
static auto HandleBraceExpressionParameterError(Context& context,
                                                Context::StateStackEntry state,
                                                State param_finish_state)
    -> void {
  bool is_type =
      param_finish_state == State::BraceExpressionParameterFinishAsType;
  bool is_value =
      param_finish_state == State::BraceExpressionParameterFinishAsValue;
  bool is_unknown =
      param_finish_state == State::BraceExpressionParameterFinishAsUnknown;
  CARBON_CHECK(is_type || is_value || is_unknown);
  CARBON_DIAGNOSTIC(ExpectedStructLiteralField, Error, "Expected {0}{1}{2}.",
                    llvm::StringRef, llvm::StringRef, llvm::StringRef);
  context.emitter().Emit(*context.position(), ExpectedStructLiteralField,
                         (is_type || is_unknown) ? "`.field: field_type`" : "",
                         is_unknown ? " or " : "",
                         (is_value || is_unknown) ? "`.field = value`" : "");

  state.state = param_finish_state;
  state.has_error = true;
  context.PushState(state);
}

// Handles BraceExpressionParameterAs(Type|Value|Unknown).
static auto HandleBraceExpressionParameter(Context& context,
                                           State after_designator_state,
                                           State param_finish_state) -> void {
  auto state = context.PopState();

  if (!context.PositionIs(Lex::TokenKind::Period)) {
    HandleBraceExpressionParameterError(context, state, param_finish_state);
    return;
  }

  state.state = after_designator_state;
  context.PushState(state);
  context.PushState(State::PeriodAsStruct);
}

auto HandleBraceExpressionParameterAsType(Context& context) -> void {
  HandleBraceExpressionParameter(
      context, State::BraceExpressionParameterAfterDesignatorAsType,
      State::BraceExpressionParameterFinishAsType);
}

auto HandleBraceExpressionParameterAsValue(Context& context) -> void {
  HandleBraceExpressionParameter(
      context, State::BraceExpressionParameterAfterDesignatorAsValue,
      State::BraceExpressionParameterFinishAsValue);
}

auto HandleBraceExpressionParameterAsUnknown(Context& context) -> void {
  HandleBraceExpressionParameter(
      context, State::BraceExpressionParameterAfterDesignatorAsUnknown,
      State::BraceExpressionParameterFinishAsUnknown);
}

// Handles BraceExpressionParameterAfterDesignatorAs(Type|Value|Unknown).
static auto HandleBraceExpressionParameterAfterDesignator(
    Context& context, State param_finish_state) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    auto recovery_pos = context.FindNextOf(
        {Lex::TokenKind::Equal, Lex::TokenKind::Colon, Lex::TokenKind::Comma});
    if (!recovery_pos ||
        context.tokens().GetKind(*recovery_pos) == Lex::TokenKind::Comma) {
      state.state = param_finish_state;
      context.PushState(state);
      return;
    }
    context.SkipTo(*recovery_pos);
  }

  // Work out the kind of this element.
  bool is_type;
  if (context.PositionIs(Lex::TokenKind::Colon)) {
    is_type = true;
  } else if (context.PositionIs(Lex::TokenKind::Equal)) {
    is_type = false;
  } else {
    HandleBraceExpressionParameterError(context, state, param_finish_state);
    return;
  }

  // If we're changing from unknown, update the related finish states.
  if (param_finish_state == State::BraceExpressionParameterFinishAsUnknown) {
    auto finish_state = context.PopState();
    CARBON_CHECK(finish_state.state == State::BraceExpressionFinishAsUnknown);
    if (is_type) {
      finish_state.state = State::BraceExpressionFinishAsType;
      param_finish_state = State::BraceExpressionParameterFinishAsType;
    } else {
      finish_state.state = State::BraceExpressionFinishAsValue;
      param_finish_state = State::BraceExpressionParameterFinishAsValue;
    }
    context.PushState(finish_state);
  }

  auto want_param_finish_state =
      is_type ? State::BraceExpressionParameterFinishAsType
              : State::BraceExpressionParameterFinishAsValue;
  if (param_finish_state != want_param_finish_state) {
    HandleBraceExpressionParameterError(context, state, param_finish_state);
    return;
  }

  // Struct type fields and value fields use the same grammar except
  // that one has a `:` separator and the other has an `=` separator.
  state.state = param_finish_state;
  state.token = context.Consume();
  context.PushState(state);
  context.PushState(State::Expression);
}

auto HandleBraceExpressionParameterAfterDesignatorAsType(Context& context)
    -> void {
  HandleBraceExpressionParameterAfterDesignator(
      context, State::BraceExpressionParameterFinishAsType);
}

auto HandleBraceExpressionParameterAfterDesignatorAsValue(Context& context)
    -> void {
  HandleBraceExpressionParameterAfterDesignator(
      context, State::BraceExpressionParameterFinishAsValue);
}

auto HandleBraceExpressionParameterAfterDesignatorAsUnknown(Context& context)
    -> void {
  HandleBraceExpressionParameterAfterDesignator(
      context, State::BraceExpressionParameterFinishAsUnknown);
}

// Handles BraceExpressionParameterFinishAs(Type|Value|Unknown).
static auto HandleBraceExpressionParameterFinish(Context& context,
                                                 NodeKind node_kind,
                                                 State param_state) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    context.AddLeafNode(NodeKind::StructFieldUnknown, state.token,
                        /*has_error=*/true);
  } else {
    context.AddNode(node_kind, state.token, state.subtree_start,
                    /*has_error=*/false);
  }

  if (context.ConsumeListToken(
          NodeKind::StructComma, Lex::TokenKind::CloseCurlyBrace,
          state.has_error) == Context::ListTokenKind::Comma) {
    context.PushState(param_state);
  }
}

auto HandleBraceExpressionParameterFinishAsType(Context& context) -> void {
  HandleBraceExpressionParameterFinish(context, NodeKind::StructFieldType,
                                       State::BraceExpressionParameterAsType);
}

auto HandleBraceExpressionParameterFinishAsValue(Context& context) -> void {
  HandleBraceExpressionParameterFinish(context, NodeKind::StructFieldValue,
                                       State::BraceExpressionParameterAsValue);
}

auto HandleBraceExpressionParameterFinishAsUnknown(Context& context) -> void {
  HandleBraceExpressionParameterFinish(
      context, NodeKind::StructFieldUnknown,
      State::BraceExpressionParameterAsUnknown);
}

// Handles BraceExpressionFinishAs(Type|Value|Unknown).
static auto HandleBraceExpressionFinish(Context& context, NodeKind node_kind)
    -> void {
  auto state = context.PopState();

  context.AddNode(node_kind, context.Consume(), state.subtree_start,
                  state.has_error);
}

auto HandleBraceExpressionFinishAsType(Context& context) -> void {
  HandleBraceExpressionFinish(context, NodeKind::StructTypeLiteral);
}

auto HandleBraceExpressionFinishAsValue(Context& context) -> void {
  HandleBraceExpressionFinish(context, NodeKind::StructLiteral);
}

auto HandleBraceExpressionFinishAsUnknown(Context& context) -> void {
  HandleBraceExpressionFinish(context, NodeKind::StructLiteral);
}

}  // namespace Carbon::Parse
