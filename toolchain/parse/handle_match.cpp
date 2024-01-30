// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

auto HandleMatchIntroducer(Context& context) -> void {
  auto state = context.PopState();
  context.AddLeafNode(NodeKind::MatchIntroducer, context.Consume());
  context.PushState(state, State::MatchConditionFinish);
  context.PushState(State::ParenConditionAsMatch);
}

auto HandleMatchConditionFinish(Context& context) -> void {
  auto state = context.PopState();
  context.PushState(state, State::MatchStatementStart);
}

auto HandleMatchStatementStart(Context& context) -> void {
  auto state = context.PopState();

  if (!context.PositionIs(Lex::TokenKind::OpenCurlyBrace)) {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(ExpectedMatchCases, Error, "Match cases expected.");
      context.emitter().Emit(*context.position(), ExpectedMatchCases);
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
  context.PushState(state, State::MatchStatementFinish);

  if (!context.PositionIs(Lex::TokenKind::CloseCurlyBrace)) {
    if (context.PositionIs(Lex::TokenKind::Case)) {
      context.PushState(State::MatchCaseIntroducer);
      context.ConsumeAndDiscard();
    } else if (context.PositionIs(Lex::TokenKind::Default)) {
      context.PushState(State::MatchDefaultStart);
      context.ConsumeAndDiscard();
    }
  }
}

auto HandleMatchCaseIntroducer(Context& context) -> void {
  auto state = context.PopState();
  context.PushState(state, State::MatchCaseAfterPattern);
  context.PushState(State::Pattern);
}

auto HandleMatchCaseAfterPattern(Context& context) -> void {
  auto state = context.PopState();
  context.PushState(state, State::MatchCaseStart);
  if (context.PositionIs(Lex::TokenKind::If)) {
    context.PushState(State::StatementIf);
  }
}

auto HandleMatchCaseStart(Context& context) -> void {
  auto state = context.PopState();
  context.AddNode(NodeKind::MatchCaseStart, context.Consume(),
                  state.subtree_start, state.has_error);
  context.PushState(state, State::MatchCaseFinish);
  context.PushState(State::CodeBlock);
}

auto HandleMatchCaseFinish(Context& context) -> void {
  auto state = context.PopState();
  context.AddNode(NodeKind::MatchCase, state.token, state.subtree_start,
                  state.has_error);
  if (context.PositionIs(Lex::TokenKind::Case)) {
    context.PushState(State::MatchCaseIntroducer);
    context.ConsumeAndDiscard();
  } else if (context.PositionIs(Lex::TokenKind::Default)) {
    context.PushState(State::MatchDefaultStart);
    context.ConsumeAndDiscard();
  }
}

auto HandleMatchDefaultStart(Context& context) -> void {
  auto state = context.PopState();
  context.AddLeafNode(NodeKind::MatchDefaultStart, context.Consume());
  context.PushState(state, State::MatchDefaultFinish);
  context.PushState(State::CodeBlock);
}

auto HandleMatchDefaultFinish(Context& context) -> void {
  auto state = context.PopState();
  context.AddNode(NodeKind::MatchDefault, state.token, state.subtree_start,
                  state.has_error);
}

auto HandleMatchStatementFinish(Context& context) -> void {
  auto state = context.PopState();
  context.AddNode(NodeKind::MatchStatement, context.Consume(),
                  state.subtree_start, state.has_error);
}

}  // namespace Carbon::Parse
