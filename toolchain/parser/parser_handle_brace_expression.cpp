// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_context.h"

namespace Carbon {

auto ParserHandleBraceExpression(ParserContext& context) -> void {
  auto state = context.PopState();

  state.state = ParserState::BraceExpressionFinishAsUnknown;
  context.PushState(state);

  CARBON_CHECK(context.ConsumeAndAddLeafNodeIf(
      TokenKind::OpenCurlyBrace,
      ParseNodeKind::StructLiteralOrStructTypeLiteralStart));
  if (!context.PositionIs(TokenKind::CloseCurlyBrace)) {
    context.PushState(ParserState::BraceExpressionParameterAsUnknown);
  }
}

// Prints a diagnostic for brace expression syntax errors.
static auto ParserHandleBraceExpressionParameterError(
    ParserContext& context, ParserContext::StateStackEntry state,
    ParserState param_finish_state) -> void {
  bool is_type =
      param_finish_state == ParserState::BraceExpressionParameterFinishAsType;
  bool is_value =
      param_finish_state == ParserState::BraceExpressionParameterFinishAsValue;
  bool is_unknown = param_finish_state ==
                    ParserState::BraceExpressionParameterFinishAsUnknown;
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
static auto ParserHandleBraceExpressionParameter(
    ParserContext& context, ParserState after_designator_state,
    ParserState param_finish_state) -> void {
  auto state = context.PopState();

  if (!context.PositionIs(TokenKind::Period)) {
    ParserHandleBraceExpressionParameterError(context, state,
                                              param_finish_state);
    return;
  }

  state.state = after_designator_state;
  context.PushState(state);
  context.PushState(ParserState::DesignatorAsStruct);
}

auto ParserHandleBraceExpressionParameterAsType(ParserContext& context)
    -> void {
  ParserHandleBraceExpressionParameter(
      context, ParserState::BraceExpressionParameterAfterDesignatorAsType,
      ParserState::BraceExpressionParameterFinishAsType);
}

auto ParserHandleBraceExpressionParameterAsValue(ParserContext& context)
    -> void {
  ParserHandleBraceExpressionParameter(
      context, ParserState::BraceExpressionParameterAfterDesignatorAsValue,
      ParserState::BraceExpressionParameterFinishAsValue);
}

auto ParserHandleBraceExpressionParameterAsUnknown(ParserContext& context)
    -> void {
  ParserHandleBraceExpressionParameter(
      context, ParserState::BraceExpressionParameterAfterDesignatorAsUnknown,
      ParserState::BraceExpressionParameterFinishAsUnknown);
}

// Handles BraceExpressionParameterAfterDesignatorAs(Type|Value|Unknown).
static auto ParserHandleBraceExpressionParameterAfterDesignator(
    ParserContext& context, ParserState param_finish_state) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    auto recovery_pos = context.FindNextOf(
        {TokenKind::Equal, TokenKind::Colon, TokenKind::Comma});
    if (!recovery_pos ||
        context.tokens().GetKind(*recovery_pos) == TokenKind::Comma) {
      state.state = param_finish_state;
      context.PushState(state);
      return;
    }
    context.SkipTo(*recovery_pos);
  }

  // Work out the kind of this element.
  bool is_type;
  if (context.PositionIs(TokenKind::Colon)) {
    is_type = true;
  } else if (context.PositionIs(TokenKind::Equal)) {
    is_type = false;
  } else {
    ParserHandleBraceExpressionParameterError(context, state,
                                              param_finish_state);
    return;
  }

  // If we're changing from unknown, update the related finish states.
  if (param_finish_state ==
      ParserState::BraceExpressionParameterFinishAsUnknown) {
    auto finish_state = context.PopState();
    CARBON_CHECK(finish_state.state ==
                 ParserState::BraceExpressionFinishAsUnknown);
    if (is_type) {
      finish_state.state = ParserState::BraceExpressionFinishAsType;
      param_finish_state = ParserState::BraceExpressionParameterFinishAsType;
    } else {
      finish_state.state = ParserState::BraceExpressionFinishAsValue;
      param_finish_state = ParserState::BraceExpressionParameterFinishAsValue;
    }
    context.PushState(finish_state);
  }

  auto want_param_finish_state =
      is_type ? ParserState::BraceExpressionParameterFinishAsType
              : ParserState::BraceExpressionParameterFinishAsValue;
  if (param_finish_state != want_param_finish_state) {
    ParserHandleBraceExpressionParameterError(context, state,
                                              param_finish_state);
    return;
  }

  // Struct type fields and value fields use the same grammar except
  // that one has a `:` separator and the other has an `=` separator.
  state.state = param_finish_state;
  state.token = context.Consume();
  context.PushState(state);
  context.PushState(ParserState::Expression);
}

auto ParserHandleBraceExpressionParameterAfterDesignatorAsType(
    ParserContext& context) -> void {
  ParserHandleBraceExpressionParameterAfterDesignator(
      context, ParserState::BraceExpressionParameterFinishAsType);
}

auto ParserHandleBraceExpressionParameterAfterDesignatorAsValue(
    ParserContext& context) -> void {
  ParserHandleBraceExpressionParameterAfterDesignator(
      context, ParserState::BraceExpressionParameterFinishAsValue);
}

auto ParserHandleBraceExpressionParameterAfterDesignatorAsUnknown(
    ParserContext& context) -> void {
  ParserHandleBraceExpressionParameterAfterDesignator(
      context, ParserState::BraceExpressionParameterFinishAsUnknown);
}

// Handles BraceExpressionParameterFinishAs(Type|Value|Unknown).
static auto ParserHandleBraceExpressionParameterFinish(ParserContext& context,
                                                       ParseNodeKind node_kind,
                                                       ParserState param_state)
    -> void {
  auto state = context.PopState();

  if (state.has_error) {
    context.AddLeafNode(ParseNodeKind::StructFieldUnknown, state.token,
                        /*has_error=*/true);
  } else {
    context.AddNode(node_kind, state.token, state.subtree_start,
                    /*has_error=*/false);
  }

  if (context.ConsumeListToken(ParseNodeKind::StructComma,
                               TokenKind::CloseCurlyBrace, state.has_error) ==
      ParserContext::ListTokenKind::Comma) {
    context.PushState(param_state);
  }
}

auto ParserHandleBraceExpressionParameterFinishAsType(ParserContext& context)
    -> void {
  ParserHandleBraceExpressionParameterFinish(
      context, ParseNodeKind::StructFieldType,
      ParserState::BraceExpressionParameterAsType);
}

auto ParserHandleBraceExpressionParameterFinishAsValue(ParserContext& context)
    -> void {
  ParserHandleBraceExpressionParameterFinish(
      context, ParseNodeKind::StructFieldValue,
      ParserState::BraceExpressionParameterAsValue);
}

auto ParserHandleBraceExpressionParameterFinishAsUnknown(ParserContext& context)
    -> void {
  ParserHandleBraceExpressionParameterFinish(
      context, ParseNodeKind::StructFieldUnknown,
      ParserState::BraceExpressionParameterAsUnknown);
}

// Handles BraceExpressionFinishAs(Type|Value|Unknown).
static auto ParserHandleBraceExpressionFinish(ParserContext& context,
                                              ParseNodeKind node_kind) -> void {
  auto state = context.PopState();

  context.AddNode(node_kind, context.Consume(), state.subtree_start,
                  state.has_error);
}

auto ParserHandleBraceExpressionFinishAsType(ParserContext& context) -> void {
  ParserHandleBraceExpressionFinish(context, ParseNodeKind::StructTypeLiteral);
}

auto ParserHandleBraceExpressionFinishAsValue(ParserContext& context) -> void {
  ParserHandleBraceExpressionFinish(context, ParseNodeKind::StructLiteral);
}

auto ParserHandleBraceExpressionFinishAsUnknown(ParserContext& context)
    -> void {
  ParserHandleBraceExpressionFinish(context, ParseNodeKind::StructLiteral);
}

}  // namespace Carbon
