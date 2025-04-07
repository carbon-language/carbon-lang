// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

auto HandleBraceExpr(Context& context) -> void {
  auto state = context.PopState();
  context.PushState(state, StateKind::BraceExprFinishAsUnknown);

  CARBON_CHECK(context.ConsumeAndAddLeafNodeIf(Lex::TokenKind::OpenCurlyBrace,
                                               NodeKind::Placeholder));
  if (!context.PositionIs(Lex::TokenKind::CloseCurlyBrace)) {
    context.PushState(StateKind::BraceExprParamAsUnknown);
  }
}

// Prints a diagnostic for brace expression syntax errors.
static auto HandleBraceExprParamError(Context& context, Context::State state,
                                      StateKind param_finish_state_kind)
    -> void {
  Diagnostics::IntAsSelect mode(0);
  switch (param_finish_state_kind) {
    case StateKind::BraceExprParamFinishAsType:
      mode.value = 0;
      break;
    case StateKind::BraceExprParamFinishAsValue:
      mode.value = 1;
      break;
    case StateKind::BraceExprParamFinishAsUnknown:
      mode.value = 2;
      break;
    default:
      CARBON_FATAL("Unexpected state: {0}", param_finish_state_kind);
  }
  CARBON_DIAGNOSTIC(
      ExpectedStructLiteralField, Error,
      "expected {0:=0:`.field: field_type`|"
      "=1:`.field = value`|=2:`.field: field_type` or `.field = value`}",
      Diagnostics::IntAsSelect);
  context.emitter().Emit(*context.position(), ExpectedStructLiteralField, mode);

  state.has_error = true;
  context.PushState(state, param_finish_state_kind);
}

// Handles BraceExprParamAs(Type|Value|Unknown).
static auto HandleBraceExprParam(Context& context,
                                 StateKind after_designator_kind,
                                 StateKind param_finish_kind) -> void {
  auto state = context.PopState();

  if (!context.PositionIs(Lex::TokenKind::Period)) {
    HandleBraceExprParamError(context, state, param_finish_kind);
    return;
  }

  context.PushState(state, after_designator_kind);
  context.PushState(StateKind::PeriodAsStruct);
}

auto HandleBraceExprParamAsType(Context& context) -> void {
  HandleBraceExprParam(context, StateKind::BraceExprParamAfterDesignatorAsType,
                       StateKind::BraceExprParamFinishAsType);
}

auto HandleBraceExprParamAsValue(Context& context) -> void {
  HandleBraceExprParam(context, StateKind::BraceExprParamAfterDesignatorAsValue,
                       StateKind::BraceExprParamFinishAsValue);
}

auto HandleBraceExprParamAsUnknown(Context& context) -> void {
  HandleBraceExprParam(context,
                       StateKind::BraceExprParamAfterDesignatorAsUnknown,
                       StateKind::BraceExprParamFinishAsUnknown);
}

// Handles BraceExprParamAfterDesignatorAs(Type|Value|Unknown).
static auto HandleBraceExprParamAfterDesignator(Context& context,
                                                StateKind param_finish_kind)
    -> void {
  auto state = context.PopState();

  if (state.has_error) {
    auto recovery_pos = context.FindNextOf(
        {Lex::TokenKind::Equal, Lex::TokenKind::Colon, Lex::TokenKind::Comma});
    if (!recovery_pos ||
        context.tokens().GetKind(*recovery_pos) == Lex::TokenKind::Comma) {
      context.PushState(state, param_finish_kind);
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
    HandleBraceExprParamError(context, state, param_finish_kind);
    return;
  }

  // If we're changing from unknown, update the related finish states.
  if (param_finish_kind == StateKind::BraceExprParamFinishAsUnknown) {
    auto finish_state = context.PopState();
    CARBON_CHECK(finish_state.kind == StateKind::BraceExprFinishAsUnknown);
    if (is_type) {
      finish_state.kind = StateKind::BraceExprFinishAsType;
      param_finish_kind = StateKind::BraceExprParamFinishAsType;
    } else {
      finish_state.kind = StateKind::BraceExprFinishAsValue;
      param_finish_kind = StateKind::BraceExprParamFinishAsValue;
    }
    context.PushState(finish_state);
  }

  auto want_param_finish_kind = is_type
                                    ? StateKind::BraceExprParamFinishAsType
                                    : StateKind::BraceExprParamFinishAsValue;
  if (param_finish_kind != want_param_finish_kind) {
    HandleBraceExprParamError(context, state, param_finish_kind);
    return;
  }

  // Struct type fields and value fields use the same grammar except
  // that one has a `:` separator and the other has an `=` separator.
  state.token = context.Consume();
  context.PushState(state, param_finish_kind);
  context.PushState(StateKind::Expr);
}

auto HandleBraceExprParamAfterDesignatorAsType(Context& context) -> void {
  HandleBraceExprParamAfterDesignator(context,
                                      StateKind::BraceExprParamFinishAsType);
}

auto HandleBraceExprParamAfterDesignatorAsValue(Context& context) -> void {
  HandleBraceExprParamAfterDesignator(context,
                                      StateKind::BraceExprParamFinishAsValue);
}

auto HandleBraceExprParamAfterDesignatorAsUnknown(Context& context) -> void {
  HandleBraceExprParamAfterDesignator(context,
                                      StateKind::BraceExprParamFinishAsUnknown);
}

// Handles BraceExprParamFinishAs(Type|Value|Unknown).
static auto HandleBraceExprParamFinish(Context& context, NodeKind field_kind,
                                       NodeKind comma_kind,
                                       StateKind param_state) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    context.AddLeafNode(NodeKind::InvalidParse, state.token,
                        /*has_error=*/true);
    context.ReturnErrorOnState();
  } else {
    context.AddNode(field_kind, state.token, /*has_error=*/false);
  }

  if (context.ConsumeListToken(comma_kind, Lex::TokenKind::CloseCurlyBrace,
                               state.has_error) ==
      Context::ListTokenKind::Comma) {
    context.PushState(param_state);
  }
}

auto HandleBraceExprParamFinishAsType(Context& context) -> void {
  HandleBraceExprParamFinish(context, NodeKind::StructTypeLiteralField,
                             NodeKind::StructTypeLiteralComma,
                             StateKind::BraceExprParamAsType);
}

auto HandleBraceExprParamFinishAsValue(Context& context) -> void {
  HandleBraceExprParamFinish(context, NodeKind::StructLiteralField,
                             NodeKind::StructLiteralComma,
                             StateKind::BraceExprParamAsValue);
}

auto HandleBraceExprParamFinishAsUnknown(Context& context) -> void {
  HandleBraceExprParamFinish(context, NodeKind::InvalidParse,
                             NodeKind::InvalidParse,
                             StateKind::BraceExprParamAsUnknown);
}

// Handles BraceExprFinishAs(Type|Value|Unknown).
static auto HandleBraceExprFinish(Context& context, NodeKind start_kind,
                                  NodeKind end_kind) -> void {
  auto state = context.PopState();

  context.ReplacePlaceholderNode(state.subtree_start, start_kind, state.token);
  context.AddNode(end_kind, context.Consume(), state.has_error);
}

auto HandleBraceExprFinishAsType(Context& context) -> void {
  HandleBraceExprFinish(context, NodeKind::StructTypeLiteralStart,
                        NodeKind::StructTypeLiteral);
}

auto HandleBraceExprFinishAsValue(Context& context) -> void {
  HandleBraceExprFinish(context, NodeKind::StructLiteralStart,
                        NodeKind::StructLiteral);
}

auto HandleBraceExprFinishAsUnknown(Context& context) -> void {
  HandleBraceExprFinishAsValue(context);
}

}  // namespace Carbon::Parse
