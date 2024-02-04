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
  }
}

auto HandleMatchCaseLoopAfterDefault(Context& context) -> void {
  context.PopAndDiscardState();

  std::optional<std::string> kind;
  if (context.PositionIs(Lex::TokenKind::Case)) {
    kind = "case";
  } else if (context.PositionIs(Lex::TokenKind::Default)) {
    kind = "default";
  }
  if (kind) {
    CARBON_DIAGNOSTIC(UnreachableMatchCase, Error, "Unreachable case; {0}",
                      std::string);
    context.emitter().Emit(
        *context.position(), UnreachableMatchCase,
        std::string{"`"} + *kind + "` occurs after the `default`");

    context.ReturnErrorOnState();
    context.PushState(State::MatchCaseLoopAfterDefault);
    context.SkipPastLikelyEnd(*context.position());
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
    context.AddLeafNode(NodeKind::MatchCaseGuardStart,
                        context.ConsumeChecked(Lex::TokenKind::OpenParen));
    context.PushState(State::Expr);
  }
}

auto HandleMatchCaseGuardFinish(Context& context) -> void {
  auto state = context.PopState();
  context.AddNode(NodeKind::MatchCaseGuard,
                  context.ConsumeChecked(Lex::TokenKind::CloseParen),
                  state.subtree_start, state.has_error);
}

auto HandleMatchCaseStart(Context& context) -> void {
  auto state = context.PopState();
  context.AddLeafNode(NodeKind::MatchCaseEqualGreater,
                      context.ConsumeChecked(Lex::TokenKind::EqualGreater));
  context.AddNode(NodeKind::MatchCaseStart,
                  context.ConsumeChecked(Lex::TokenKind::OpenCurlyBrace),
                  state.subtree_start, state.has_error);
  context.PushState(state, State::MatchCaseFinish);
  context.PushState(State::StatementScopeLoop);
}

auto HandleMatchCaseFinish(Context& context) -> void {
  auto state = context.PopState();
  context.AddNode(NodeKind::MatchCase,
                  context.ConsumeChecked(Lex::TokenKind::CloseCurlyBrace),
                  state.subtree_start, state.has_error);
}

auto HandleMatchDefaultIntroducer(Context& context) -> void {
  auto state = context.PopState();
  context.AddLeafNode(NodeKind::MatchDefaultIntroducer,
                      context.ConsumeChecked(Lex::TokenKind::Default));
  context.AddLeafNode(NodeKind::MatchDefaultEqualGreater,
                      context.ConsumeChecked(Lex::TokenKind::EqualGreater));
  context.AddNode(NodeKind::MatchDefaultStart,
                  context.ConsumeChecked(Lex::TokenKind::OpenCurlyBrace),
                  state.subtree_start, state.has_error);
  context.PushState(state, State::MatchDefaultFinish);
  context.PushState(State::StatementScopeLoop);
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
