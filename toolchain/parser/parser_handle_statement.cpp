// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_context.h"

namespace Carbon {

auto ParserHandleStatement(ParserContext& context) -> void {
  context.PopAndDiscardState();

  switch (context.PositionKind()) {
    case TokenKind::Break: {
      context.PushState(ParserState::StatementBreakFinish);
      context.AddLeafNode(ParseNodeKind::BreakStatementStart,
                          context.Consume());
      break;
    }
    case TokenKind::Continue: {
      context.PushState(ParserState::StatementContinueFinish);
      context.AddLeafNode(ParseNodeKind::ContinueStatementStart,
                          context.Consume());
      break;
    }
    case TokenKind::For: {
      context.PushState(ParserState::StatementForFinish);
      context.PushState(ParserState::StatementForHeader);
      ++context.position();
      break;
    }
    case TokenKind::If: {
      context.PushState(ParserState::StatementIf);
      break;
    }
    case TokenKind::Return: {
      context.PushState(ParserState::StatementReturn);
      break;
    }
    case TokenKind::Var: {
      context.PushState(ParserState::VarAsSemicolon);
      break;
    }
    case TokenKind::While: {
      context.PushState(ParserState::StatementWhile);
      break;
    }
    default: {
      context.PushState(ParserState::ExpressionStatementFinish);
      context.PushState(ParserState::Expression);
      break;
    }
  }
}

// Handles the `;` after a keyword statement.
static auto ParserHandleStatementKeywordFinish(ParserContext& context,
                                               ParseNodeKind node_kind)
    -> void {
  auto state = context.PopState();

  auto semi = context.ConsumeIf(TokenKind::Semi);
  if (!semi) {
    CARBON_DIAGNOSTIC(ExpectedSemiAfter, Error, "Expected `;` after `{0}`.",
                      TokenKind);
    context.emitter().Emit(*context.position(), ExpectedSemiAfter,
                           context.tokens().GetKind(state.token));
    state.has_error = true;
    // Recover to the next semicolon if possible, otherwise indicate the
    // keyword for the error.
    semi = context.SkipPastLikelyEnd(state.token);
    if (!semi) {
      semi = state.token;
    }
  }
  context.AddNode(node_kind, *semi, state.subtree_start, state.has_error);
}

auto ParserHandleStatementBreakFinish(ParserContext& context) -> void {
  ParserHandleStatementKeywordFinish(context, ParseNodeKind::BreakStatement);
}

auto ParserHandleStatementContinueFinish(ParserContext& context) -> void {
  ParserHandleStatementKeywordFinish(context, ParseNodeKind::ContinueStatement);
}

auto ParserHandleStatementForHeader(ParserContext& context) -> void {
  auto state = context.PopState();

  context.ConsumeAndAddOpenParen(state.token, ParseNodeKind::ForHeaderStart);

  state.state = ParserState::StatementForHeaderIn;

  if (context.PositionIs(TokenKind::Var)) {
    context.PushState(state);
    context.PushState(ParserState::VarAsFor);
  } else {
    CARBON_DIAGNOSTIC(ExpectedVariableDeclaration, Error,
                      "Expected `var` declaration.");
    context.emitter().Emit(*context.position(), ExpectedVariableDeclaration);

    if (auto next_in = context.FindNextOf({TokenKind::In})) {
      context.SkipTo(*next_in);
      ++context.position();
    }
    state.has_error = true;
    context.PushState(state);
  }
}

auto ParserHandleStatementForHeaderIn(ParserContext& context) -> void {
  auto state = context.PopState();

  state.state = ParserState::StatementForHeaderFinish;
  context.PushState(state);
  context.PushState(ParserState::Expression);
}

auto ParserHandleStatementForHeaderFinish(ParserContext& context) -> void {
  auto state = context.PopState();

  context.ConsumeAndAddCloseParen(state, ParseNodeKind::ForHeader);

  context.PushState(ParserState::CodeBlock);
}

auto ParserHandleStatementForFinish(ParserContext& context) -> void {
  auto state = context.PopState();

  context.AddNode(ParseNodeKind::ForStatement, state.token, state.subtree_start,
                  state.has_error);
}

auto ParserHandleStatementIf(ParserContext& context) -> void {
  context.PopAndDiscardState();

  context.PushState(ParserState::StatementIfConditionFinish);
  context.PushState(ParserState::ParenConditionAsIf);
  ++context.position();
}

auto ParserHandleStatementIfConditionFinish(ParserContext& context) -> void {
  auto state = context.PopState();

  state.state = ParserState::StatementIfThenBlockFinish;
  context.PushState(state);
  context.PushState(ParserState::CodeBlock);
}

auto ParserHandleStatementIfThenBlockFinish(ParserContext& context) -> void {
  auto state = context.PopState();

  if (context.ConsumeAndAddLeafNodeIf(TokenKind::Else,
                                      ParseNodeKind::IfStatementElse)) {
    state.state = ParserState::StatementIfElseBlockFinish;
    context.PushState(state);
    // `else if` is permitted as a special case.
    context.PushState(context.PositionIs(TokenKind::If)
                          ? ParserState::StatementIf
                          : ParserState::CodeBlock);
  } else {
    context.AddNode(ParseNodeKind::IfStatement, state.token,
                    state.subtree_start, state.has_error);
  }
}

auto ParserHandleStatementIfElseBlockFinish(ParserContext& context) -> void {
  auto state = context.PopState();
  context.AddNode(ParseNodeKind::IfStatement, state.token, state.subtree_start,
                  state.has_error);
}

auto ParserHandleStatementReturn(ParserContext& context) -> void {
  auto state = context.PopState();
  state.state = ParserState::StatementReturnFinish;
  context.PushState(state);

  context.AddLeafNode(ParseNodeKind::ReturnStatementStart, context.Consume());
  if (!context.PositionIs(TokenKind::Semi)) {
    context.PushState(ParserState::Expression);
  }
}

auto ParserHandleStatementReturnFinish(ParserContext& context) -> void {
  ParserHandleStatementKeywordFinish(context, ParseNodeKind::ReturnStatement);
}

auto ParserHandleStatementScopeLoop(ParserContext& context) -> void {
  // This maintains the current state until we're at the end of the scope.

  auto token_kind = context.PositionKind();
  if (token_kind == TokenKind::CloseCurlyBrace) {
    auto state = context.PopState();
    if (state.has_error) {
      context.ReturnErrorOnState();
    }
  } else {
    context.PushState(ParserState::Statement);
  }
}

auto ParserHandleStatementWhile(ParserContext& context) -> void {
  context.PopAndDiscardState();

  context.PushState(ParserState::StatementWhileConditionFinish);
  context.PushState(ParserState::ParenConditionAsWhile);
  ++context.position();
}

auto ParserHandleStatementWhileConditionFinish(ParserContext& context) -> void {
  auto state = context.PopState();

  state.state = ParserState::StatementWhileBlockFinish;
  context.PushState(state);
  context.PushState(ParserState::CodeBlock);
}

auto ParserHandleStatementWhileBlockFinish(ParserContext& context) -> void {
  auto state = context.PopState();

  context.AddNode(ParseNodeKind::WhileStatement, state.token,
                  state.subtree_start, state.has_error);
}

}  // namespace Carbon
