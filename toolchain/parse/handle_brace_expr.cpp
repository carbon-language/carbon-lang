// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

auto HandleBraceExpr(Context& context) -> void {
  auto state = context.PopState();

  state.state = State::BraceExprFinishAsUnknown;
  context.PushState(state);

  CARBON_CHECK(context.ConsumeAndAddLeafNodeIf(
      Lex::TokenKind::OpenCurlyBrace,
      NodeKind::StructLiteralOrStructTypeLiteralStart));
  if (!context.PositionIs(Lex::TokenKind::CloseCurlyBrace)) {
    context.PushState(State::BraceExprParameterAsUnknown);
  }
}

// Prints a diagnostic for brace expression syntax errors.
static auto HandleBraceExprParameterError(Context& context,
                                          Context::StateStackEntry state,
                                          State param_finish_state) -> void {
  bool is_type = param_finish_state == State::BraceExprParameterFinishAsType;
  bool is_value = param_finish_state == State::BraceExprParameterFinishAsValue;
  bool is_unknown =
      param_finish_state == State::BraceExprParameterFinishAsUnknown;
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

// Handles BraceExprParameterAs(Type|Value|Unknown).
static auto HandleBraceExprParameter(Context& context,
                                     State after_designator_state,
                                     State param_finish_state) -> void {
  auto state = context.PopState();

  if (!context.PositionIs(Lex::TokenKind::Period)) {
    HandleBraceExprParameterError(context, state, param_finish_state);
    return;
  }

  state.state = after_designator_state;
  context.PushState(state);
  context.PushState(State::PeriodAsStruct);
}

auto HandleBraceExprParameterAsType(Context& context) -> void {
  HandleBraceExprParameter(context,
                           State::BraceExprParameterAfterDesignatorAsType,
                           State::BraceExprParameterFinishAsType);
}

auto HandleBraceExprParameterAsValue(Context& context) -> void {
  HandleBraceExprParameter(context,
                           State::BraceExprParameterAfterDesignatorAsValue,
                           State::BraceExprParameterFinishAsValue);
}

auto HandleBraceExprParameterAsUnknown(Context& context) -> void {
  HandleBraceExprParameter(context,
                           State::BraceExprParameterAfterDesignatorAsUnknown,
                           State::BraceExprParameterFinishAsUnknown);
}

// Handles BraceExprParameterAfterDesignatorAs(Type|Value|Unknown).
static auto HandleBraceExprParameterAfterDesignator(Context& context,
                                                    State param_finish_state)
    -> void {
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
    HandleBraceExprParameterError(context, state, param_finish_state);
    return;
  }

  // If we're changing from unknown, update the related finish states.
  if (param_finish_state == State::BraceExprParameterFinishAsUnknown) {
    auto finish_state = context.PopState();
    CARBON_CHECK(finish_state.state == State::BraceExprFinishAsUnknown);
    if (is_type) {
      finish_state.state = State::BraceExprFinishAsType;
      param_finish_state = State::BraceExprParameterFinishAsType;
    } else {
      finish_state.state = State::BraceExprFinishAsValue;
      param_finish_state = State::BraceExprParameterFinishAsValue;
    }
    context.PushState(finish_state);
  }

  auto want_param_finish_state = is_type
                                     ? State::BraceExprParameterFinishAsType
                                     : State::BraceExprParameterFinishAsValue;
  if (param_finish_state != want_param_finish_state) {
    HandleBraceExprParameterError(context, state, param_finish_state);
    return;
  }

  // Struct type fields and value fields use the same grammar except
  // that one has a `:` separator and the other has an `=` separator.
  state.state = param_finish_state;
  state.token = context.Consume();
  context.PushState(state);
  context.PushState(State::Expr);
}

auto HandleBraceExprParameterAfterDesignatorAsType(Context& context) -> void {
  HandleBraceExprParameterAfterDesignator(
      context, State::BraceExprParameterFinishAsType);
}

auto HandleBraceExprParameterAfterDesignatorAsValue(Context& context) -> void {
  HandleBraceExprParameterAfterDesignator(
      context, State::BraceExprParameterFinishAsValue);
}

auto HandleBraceExprParameterAfterDesignatorAsUnknown(Context& context)
    -> void {
  HandleBraceExprParameterAfterDesignator(
      context, State::BraceExprParameterFinishAsUnknown);
}

// Handles BraceExprParameterFinishAs(Type|Value|Unknown).
static auto HandleBraceExprParameterFinish(Context& context, NodeKind node_kind,
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

auto HandleBraceExprParameterFinishAsType(Context& context) -> void {
  HandleBraceExprParameterFinish(context, NodeKind::StructFieldType,
                                 State::BraceExprParameterAsType);
}

auto HandleBraceExprParameterFinishAsValue(Context& context) -> void {
  HandleBraceExprParameterFinish(context, NodeKind::StructFieldValue,
                                 State::BraceExprParameterAsValue);
}

auto HandleBraceExprParameterFinishAsUnknown(Context& context) -> void {
  HandleBraceExprParameterFinish(context, NodeKind::StructFieldUnknown,
                                 State::BraceExprParameterAsUnknown);
}

// Handles BraceExprFinishAs(Type|Value|Unknown).
static auto HandleBraceExprFinish(Context& context, NodeKind node_kind)
    -> void {
  auto state = context.PopState();

  context.AddNode(node_kind, context.Consume(), state.subtree_start,
                  state.has_error);
}

auto HandleBraceExprFinishAsType(Context& context) -> void {
  HandleBraceExprFinish(context, NodeKind::StructTypeLiteral);
}

auto HandleBraceExprFinishAsValue(Context& context) -> void {
  HandleBraceExprFinish(context, NodeKind::StructLiteral);
}

auto HandleBraceExprFinishAsUnknown(Context& context) -> void {
  HandleBraceExprFinish(context, NodeKind::StructLiteral);
}

}  // namespace Carbon::Parse
