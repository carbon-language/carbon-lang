// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

static auto HandleStatementsBlockStart(Context& context, State finish,
                                       NodeKind equal_greater, NodeKind starter,
                                       NodeKind complete) -> void {
  auto state = context.PopState();

  if (!context.PositionIs(Lex::TokenKind::EqualGreater)) {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(ExpectedMatchCaseArrow, Error,
                        "Expected `=>` introducing statement block.");
      context.emitter().Emit(*context.position(), ExpectedMatchCaseArrow);
    }

    context.AddLeafNode(equal_greater, *context.position(), true);
    context.AddNode(starter, *context.position(), state.subtree_start, true);
    context.AddNode(complete, *context.position(), state.subtree_start, true);
    context.SkipPastLikelyEnd(*context.position());
    return;
  }

  context.AddLeafNode(equal_greater, context.Consume());

  if (!context.PositionIs(Lex::TokenKind::OpenCurlyBrace)) {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(ExpectedMatchCaseBlock, Error,
                        "Expected `{{` after `=>`.");
      context.emitter().Emit(*context.position(), ExpectedMatchCaseBlock);
    }

    context.AddNode(starter, *context.position(), state.subtree_start, true);
    context.AddNode(complete, *context.position(), state.subtree_start, true);
    context.SkipPastLikelyEnd(*context.position());
    return;
  }

  context.AddNode(starter, context.Consume(), state.subtree_start,
                  state.has_error);
  context.PushState(state, finish);
  context.PushState(State::StatementScopeLoop);
}

static auto EmitUnexpectedTokenAndRecover(Context& context) -> void {
  CARBON_DIAGNOSTIC(UnexpectedTokenInMatchCasesBlock, Error,
                    "Unexpected `{0}`; expected `case`, `default` or `}`.",
                    Lex::TokenKind);
  context.emitter().Emit(*context.position(), UnexpectedTokenInMatchCasesBlock,
                         context.PositionKind());
  context.ReturnErrorOnState();
  context.SkipPastLikelyEnd(*context.position());
}

auto HandleMatchIntroducer(Context& context) -> void {
  auto state = context.PopState();
  context.AddLeafNode(NodeKind::Placeholder, *context.position());
  context.PushState(state, State::MatchConditionFinish);
  context.PushState(State::ParenConditionAsMatch);
  context.ConsumeAndDiscard();
}

auto HandleMatchConditionFinish(Context& context) -> void {
  auto state = context.PopState();
  context.ReplacePlaceholderNode(state.subtree_start, NodeKind::MatchIntroducer,
                                 state.token);

  if (!context.PositionIs(Lex::TokenKind::OpenCurlyBrace)) {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(ExpectedMatchCasesBlock, Error,
                        "Expected `{{` starting block with cases.");
      context.emitter().Emit(*context.position(), ExpectedMatchCasesBlock);
    }

    context.AddNode(NodeKind::MatchStatementStart, *context.position(),
                    state.subtree_start, true);
    context.AddNode(NodeKind::MatchStatement, *context.position(),
                    state.subtree_start, true);
    context.SkipPastLikelyEnd(*context.position());
    return;
  }

  context.AddNode(NodeKind::MatchStatementStart, context.Consume(),
                  state.subtree_start, state.has_error);

  state.has_error = false;
  if (context.PositionIs(Lex::TokenKind::CloseCurlyBrace)) {
    CARBON_DIAGNOSTIC(ExpectedMatchCases, Error, "Expected cases.");
    context.emitter().Emit(*context.position(), ExpectedMatchCases);
    state.has_error = true;
  }

  context.PushState(state, State::MatchStatementFinish);
  context.PushState(State::MatchCaseLoop);
}

auto HandleMatchCaseLoop(Context& context) -> void {
  context.PopAndDiscardState();

  if (context.PositionIs(Lex::TokenKind::Case)) {
    context.PushState(State::MatchCaseLoop);
    context.PushState(State::MatchCaseIntroducer);
  } else if (context.PositionIs(Lex::TokenKind::Default)) {
    context.PushState(State::MatchCaseLoopAfterDefault);
    context.PushState(State::MatchDefaultIntroducer);
  } else if (!context.PositionIs(Lex::TokenKind::CloseCurlyBrace)) {
    EmitUnexpectedTokenAndRecover(context);
    context.PushState(State::MatchCaseLoop);
  }
}

auto HandleMatchCaseLoopAfterDefault(Context& context) -> void {
  context.PopAndDiscardState();

  Lex::TokenKind kind = context.PositionKind();
  if (kind == Lex::TokenKind::Case or kind == Lex::TokenKind::Default) {
    CARBON_DIAGNOSTIC(UnreachableMatchCase, Error, "Unreachable case; `{0}{1}",
                      Lex::TokenKind, std::string);
    context.emitter().Emit(*context.position(), UnreachableMatchCase, kind,
                           "` occurs after the `default`");

    context.ReturnErrorOnState();
    context.PushState(State::MatchCaseLoopAfterDefault);
    context.SkipPastLikelyEnd(*context.position());
    return;
  } else if (kind != Lex::TokenKind::CloseCurlyBrace) {
    EmitUnexpectedTokenAndRecover(context);
    context.PushState(State::MatchCaseLoopAfterDefault);
  }
}

auto HandleMatchCaseIntroducer(Context& context) -> void {
  auto state = context.PopState();

  context.AddLeafNode(NodeKind::MatchCaseIntroducer, context.Consume());
  context.PushState(state, State::MatchCaseAfterPattern);
  context.PushState(State::Pattern);
}

auto HandleMatchCaseAfterPattern(Context& context) -> void {
  auto state = context.PopState();
  if (state.has_error) {
    context.AddNode(NodeKind::MatchCaseStart, *context.position(),
                    state.subtree_start, true);
    context.AddNode(NodeKind::MatchCase, *context.position(),
                    state.subtree_start, true);
    context.SkipPastLikelyEnd(*context.position());
    return;
  }

  context.PushState(state, State::MatchCaseStart);
  if (context.PositionIs(Lex::TokenKind::If)) {
    context.PushState(State::MatchCaseGuardFinish);
    context.AddLeafNode(NodeKind::MatchCaseGuardIntroducer, context.Consume());
    auto open_paren = context.ConsumeIf(Lex::TokenKind::OpenParen);
    if (open_paren) {
      context.AddLeafNode(NodeKind::MatchCaseGuardStart, *open_paren);
      context.PushState(State::Expr);
    } else {
      context.AddLeafNode(NodeKind::MatchCaseGuardStart, *context.position(),
                          true);
      context.AddLeafNode(NodeKind::InvalidParse, *context.position(), true);
      state = context.PopState();
      context.AddNode(NodeKind::MatchCaseGuard, *context.position(),
                      state.subtree_start, true);
      state = context.PopState();
      context.AddNode(NodeKind::MatchCaseStart, *context.position(),
                      state.subtree_start, true);
      context.AddNode(NodeKind::MatchCase, *context.position(),
                      state.subtree_start, true);
      context.SkipPastLikelyEnd(*context.position());
      return;
    }
  }
}

auto HandleMatchCaseGuardFinish(Context& context) -> void {
  auto state = context.PopState();

  auto close_paren = context.ConsumeIf(Lex::TokenKind::CloseParen);
  if (close_paren) {
    context.AddNode(NodeKind::MatchCaseGuard, *close_paren, state.subtree_start,
                    state.has_error);
  } else {
    context.AddNode(NodeKind::MatchCaseGuard, *context.position(),
                    state.subtree_start, true);
    context.ReturnErrorOnState();
    context.SkipPastLikelyEnd(*context.position());
    return;
  }
}

auto HandleMatchCaseStart(Context& context) -> void {
  HandleStatementsBlockStart(context, State::MatchCaseFinish,
                             NodeKind::MatchCaseEqualGreater,
                             NodeKind::MatchCaseStart, NodeKind::MatchCase);
}

auto HandleMatchCaseFinish(Context& context) -> void {
  auto state = context.PopState();
  context.AddNode(NodeKind::MatchCase,
                  context.ConsumeChecked(Lex::TokenKind::CloseCurlyBrace),
                  state.subtree_start, state.has_error);
}

auto HandleMatchDefaultIntroducer(Context& context) -> void {
  context.AddLeafNode(NodeKind::MatchDefaultIntroducer, context.Consume());

  HandleStatementsBlockStart(
      context, State::MatchDefaultFinish, NodeKind::MatchDefaultEqualGreater,
      NodeKind::MatchDefaultStart, NodeKind::MatchDefault);
}

auto HandleMatchDefaultFinish(Context& context) -> void {
  auto state = context.PopState();
  context.AddNode(NodeKind::MatchDefault,
                  context.ConsumeChecked(Lex::TokenKind::CloseCurlyBrace),
                  state.subtree_start, state.has_error);
}

auto HandleMatchStatementFinish(Context& context) -> void {
  auto state = context.PopState();
  context.AddNode(NodeKind::MatchStatement,
                  context.ConsumeChecked(Lex::TokenKind::CloseCurlyBrace),
                  state.subtree_start, state.has_error);
}

}  // namespace Carbon::Parse
