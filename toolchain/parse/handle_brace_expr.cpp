// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

auto HandleBraceExpr(Context& context) -> void {
  auto state = context.PopState();
  context.PushState(state, State::BraceExprFinishAsUnknown);

  CARBON_CHECK(context.ConsumeAndAddLeafNodeIf(
      Lex::TokenKind::OpenCurlyBrace,
      NodeKind::StructLiteralOrStructTypeLiteralStart));
  if (!context.PositionIs(Lex::TokenKind::CloseCurlyBrace)) {
    context.PushState(State::BraceExprParamAsUnknown);
  }
}

// Prints a diagnostic for brace expression syntax errors.
static auto HandleBraceExprParamError(Context& context,
                                      Context::StateStackEntry state,
                                      State param_finish_state) -> void {
  bool is_type = param_finish_state == State::BraceExprParamFinishAsType;
  bool is_value = param_finish_state == State::BraceExprParamFinishAsValue;
  bool is_unknown = param_finish_state == State::BraceExprParamFinishAsUnknown;
  CARBON_CHECK(is_type || is_value || is_unknown);
  CARBON_DIAGNOSTIC(ExpectedStructLiteralField, Error, "Expected {0}{1}{2}.",
                    llvm::StringLiteral, llvm::StringLiteral,
                    llvm::StringLiteral);
  context.emitter().Emit(
      *context.position(), ExpectedStructLiteralField,
      (is_type || is_unknown) ? llvm::StringLiteral("`.field: field_type`")
                              : llvm::StringLiteral(""),
      is_unknown ? llvm::StringLiteral(" or ") : llvm::StringLiteral(""),
      (is_value || is_unknown) ? llvm::StringLiteral("`.field = value`")
                               : llvm::StringLiteral(""));

  state.has_error = true;
  context.PushState(state, param_finish_state);
}

// Handles BraceExprParamAs(Type|Value|Unknown).
static auto HandleBraceExprParam(Context& context, State after_designator_state,
                                 State param_finish_state) -> void {
  auto state = context.PopState();

  if (!context.PositionIs(Lex::TokenKind::Period)) {
    HandleBraceExprParamError(context, state, param_finish_state);
    return;
  }

  context.PushState(state, after_designator_state);
  context.PushState(State::PeriodAsStruct);
}

auto HandleBraceExprParamAsType(Context& context) -> void {
  HandleBraceExprParam(context, State::BraceExprParamAfterDesignatorAsType,
                       State::BraceExprParamFinishAsType);
}

auto HandleBraceExprParamAsValue(Context& context) -> void {
  HandleBraceExprParam(context, State::BraceExprParamAfterDesignatorAsValue,
                       State::BraceExprParamFinishAsValue);
}

auto HandleBraceExprParamAsUnknown(Context& context) -> void {
  HandleBraceExprParam(context, State::BraceExprParamAfterDesignatorAsUnknown,
                       State::BraceExprParamFinishAsUnknown);
}

// Handles BraceExprParamAfterDesignatorAs(Type|Value|Unknown).
static auto HandleBraceExprParamAfterDesignator(Context& context,
                                                State param_finish_state)
    -> void {
  auto state = context.PopState();

  if (state.has_error) {
    auto recovery_pos = context.FindNextOf(
        {Lex::TokenKind::Equal, Lex::TokenKind::Colon, Lex::TokenKind::Comma});
    if (!recovery_pos ||
        context.tokens().GetKind(*recovery_pos) == Lex::TokenKind::Comma) {
      context.PushState(state, param_finish_state);
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
    HandleBraceExprParamError(context, state, param_finish_state);
    return;
  }

  // If we're changing from unknown, update the related finish states.
  if (param_finish_state == State::BraceExprParamFinishAsUnknown) {
    auto finish_state = context.PopState();
    CARBON_CHECK(finish_state.state == State::BraceExprFinishAsUnknown);
    if (is_type) {
      finish_state.state = State::BraceExprFinishAsType;
      param_finish_state = State::BraceExprParamFinishAsType;
    } else {
      finish_state.state = State::BraceExprFinishAsValue;
      param_finish_state = State::BraceExprParamFinishAsValue;
    }
    context.PushState(finish_state);
  }

  auto want_param_finish_state = is_type ? State::BraceExprParamFinishAsType
                                         : State::BraceExprParamFinishAsValue;
  if (param_finish_state != want_param_finish_state) {
    HandleBraceExprParamError(context, state, param_finish_state);
    return;
  }

  // Struct type fields and value fields use the same grammar except
  // that one has a `:` separator and the other has an `=` separator.
  state.token = context.Consume();
  context.PushState(state, param_finish_state);
  context.PushState(State::Expr);
}

auto HandleBraceExprParamAfterDesignatorAsType(Context& context) -> void {
  HandleBraceExprParamAfterDesignator(context,
                                      State::BraceExprParamFinishAsType);
}

auto HandleBraceExprParamAfterDesignatorAsValue(Context& context) -> void {
  HandleBraceExprParamAfterDesignator(context,
                                      State::BraceExprParamFinishAsValue);
}

auto HandleBraceExprParamAfterDesignatorAsUnknown(Context& context) -> void {
  HandleBraceExprParamAfterDesignator(context,
                                      State::BraceExprParamFinishAsUnknown);
}

// Handles BraceExprParamFinishAs(Type|Value|Unknown).
static auto HandleBraceExprParamFinish(Context& context, NodeKind node_kind,
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

auto HandleBraceExprParamFinishAsType(Context& context) -> void {
  HandleBraceExprParamFinish(context, NodeKind::StructFieldType,
                             State::BraceExprParamAsType);
}

auto HandleBraceExprParamFinishAsValue(Context& context) -> void {
  HandleBraceExprParamFinish(context, NodeKind::StructFieldValue,
                             State::BraceExprParamAsValue);
}

auto HandleBraceExprParamFinishAsUnknown(Context& context) -> void {
  HandleBraceExprParamFinish(context, NodeKind::StructFieldUnknown,
                             State::BraceExprParamAsUnknown);
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
