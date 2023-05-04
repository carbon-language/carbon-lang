// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_context.h"

namespace Carbon {

// Handles VarAs(Semicolon|For).
static auto ParserHandleVar(ParserContext& context, ParserState finish_state)
    -> void {
  context.PopAndDiscardState();

  // These will start at the `var`.
  context.PushState(finish_state);
  context.PushState(ParserState::VarAfterPattern);

  context.AddLeafNode(ParseNodeKind::VariableIntroducer, context.Consume());

  // This will start at the pattern.
  context.PushState(ParserState::PatternAsVariable);
}

auto ParserHandleVarAsSemicolon(ParserContext& context) -> void {
  ParserHandleVar(context, ParserState::VarFinishAsSemicolon);
}

auto ParserHandleVarAsFor(ParserContext& context) -> void {
  ParserHandleVar(context, ParserState::VarFinishAsFor);
}

auto ParserHandleVarAfterPattern(ParserContext& context) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    if (auto after_pattern =
            context.FindNextOf({TokenKind::Equal, TokenKind::Semi})) {
      context.SkipTo(*after_pattern);
    }
  }

  if (auto equals = context.ConsumeIf(TokenKind::Equal)) {
    context.AddLeafNode(ParseNodeKind::VariableInitializer, *equals);
    context.PushState(ParserState::Expression);
  }
}

auto ParserHandleVarFinishAsSemicolon(ParserContext& context) -> void {
  auto state = context.PopState();

  auto end_token = state.token;
  if (context.PositionIs(TokenKind::Semi)) {
    end_token = context.Consume();
  } else {
    context.emitter().Emit(*context.position(), ExpectedSemiAfterExpression);
    state.has_error = true;
    if (auto semi_token = context.SkipPastLikelyEnd(state.token)) {
      end_token = *semi_token;
    }
  }
  context.AddNode(ParseNodeKind::VariableDeclaration, end_token,
                  state.subtree_start, state.has_error);
}

auto ParserHandleVarFinishAsFor(ParserContext& context) -> void {
  auto state = context.PopState();

  auto end_token = state.token;
  if (context.PositionIs(TokenKind::In)) {
    end_token = context.Consume();
  } else if (context.PositionIs(TokenKind::Colon)) {
    CARBON_DIAGNOSTIC(ExpectedInNotColon, Error,
                      "`:` should be replaced by `in`.");
    context.emitter().Emit(*context.position(), ExpectedInNotColon);
    state.has_error = true;
    end_token = context.Consume();
  } else {
    CARBON_DIAGNOSTIC(ExpectedIn, Error,
                      "Expected `in` after loop `var` declaration.");
    context.emitter().Emit(*context.position(), ExpectedIn);
    state.has_error = true;
  }

  context.AddNode(ParseNodeKind::ForIn, end_token, state.subtree_start,
                  state.has_error);
}

}  // namespace Carbon
