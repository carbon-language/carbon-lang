// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

static auto HandleStatementsBlockStart(Context& context, StateKind finish,
                                       NodeKind equal_greater, NodeKind starter,
                                       NodeKind complete) -> void {
  auto state = context.PopState();

  if (!context.PositionIs(Lex::TokenKind::EqualGreater)) {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(ExpectedMatchCaseArrow, Error,
                        "expected `=>` introducing statement block");
      context.emitter().Emit(*context.position(), ExpectedMatchCaseArrow);
    }

    context.AddLeafNode(equal_greater, *context.position(), /*has_error=*/true);
    context.AddNode(starter, *context.position(), /*has_error=*/true);
    context.AddNode(complete, *context.position(), /*has_error=*/true);
    context.SkipPastLikelyEnd(*context.position());
    return;
  }

  context.AddLeafNode(equal_greater, context.Consume());

  if (!context.PositionIs(Lex::TokenKind::OpenCurlyBrace)) {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(ExpectedMatchCaseBlock, Error,
                        "expected `{{` after `=>`");
      context.emitter().Emit(*context.position(), ExpectedMatchCaseBlock);
    }

    context.AddNode(starter, *context.position(), /*has_error=*/true);
    context.AddNode(complete, *context.position(), /*has_error=*/true);
    context.SkipPastLikelyEnd(*context.position());
    return;
  }

  context.AddNode(starter, context.Consume(), state.has_error);
  context.PushState(state, finish);
  context.PushState(StateKind::StatementScopeLoop);
}

static auto EmitUnexpectedTokenAndRecover(Context& context) -> void {
  CARBON_DIAGNOSTIC(UnexpectedTokenInMatchCasesBlock, Error,
                    "unexpected `{0}`; expected `case`, `default` or `}`",
                    Lex::TokenKind);
  context.emitter().Emit(*context.position(), UnexpectedTokenInMatchCasesBlock,
                         context.PositionKind());
  context.ReturnErrorOnState();
  context.SkipPastLikelyEnd(*context.position());
}

auto HandleMatchIntroducer(Context& context) -> void {
  auto state = context.PopState();
  context.AddLeafNode(NodeKind::Placeholder, *context.position());
  context.PushState(state, StateKind::MatchConditionFinish);
  context.PushState(StateKind::ParenConditionAsMatch);
  context.ConsumeAndDiscard();
}

auto HandleMatchConditionFinish(Context& context) -> void {
  auto state = context.PopState();
  context.ReplacePlaceholderNode(state.subtree_start, NodeKind::MatchIntroducer,
                                 state.token);

  if (!context.PositionIs(Lex::TokenKind::OpenCurlyBrace)) {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(ExpectedMatchCasesBlock, Error,
                        "expected `{{` starting block with cases");
      context.emitter().Emit(*context.position(), ExpectedMatchCasesBlock);
    }

    context.AddNode(NodeKind::MatchStatementStart, *context.position(),
                    /*has_error=*/true);
    context.AddNode(NodeKind::MatchStatement, *context.position(),
                    /*has_error=*/true);
    context.SkipPastLikelyEnd(*context.position());
    return;
  }

  context.AddNode(NodeKind::MatchStatementStart, context.Consume(),
                  state.has_error);

  state.has_error = false;
  if (context.PositionIs(Lex::TokenKind::CloseCurlyBrace)) {
    CARBON_DIAGNOSTIC(ExpectedMatchCases, Error, "expected cases");
    context.emitter().Emit(*context.position(), ExpectedMatchCases);
    state.has_error = true;
  }

  context.PushState(state, StateKind::MatchStatementFinish);
  context.PushState(StateKind::MatchCaseLoop);
}

auto HandleMatchCaseLoop(Context& context) -> void {
  context.PopAndDiscardState();

  if (context.PositionIs(Lex::TokenKind::Case)) {
    context.PushState(StateKind::MatchCaseLoop);
    context.PushState(StateKind::MatchCaseIntroducer);
  } else if (context.PositionIs(Lex::TokenKind::Default)) {
    context.PushState(StateKind::MatchCaseLoopAfterDefault);
    context.PushState(StateKind::MatchDefaultIntroducer);
  } else if (!context.PositionIs(Lex::TokenKind::CloseCurlyBrace)) {
    EmitUnexpectedTokenAndRecover(context);
    context.PushState(StateKind::MatchCaseLoop);
  }
}

auto HandleMatchCaseLoopAfterDefault(Context& context) -> void {
  context.PopAndDiscardState();

  Lex::TokenKind kind = context.PositionKind();
  if (kind == Lex::TokenKind::Case or kind == Lex::TokenKind::Default) {
    CARBON_DIAGNOSTIC(UnreachableMatchCase, Error,
                      "unreachable case; `{0}` occurs after the `default`",
                      Lex::TokenKind);
    context.emitter().Emit(*context.position(), UnreachableMatchCase, kind);

    context.ReturnErrorOnState();
    context.PushState(StateKind::MatchCaseLoopAfterDefault);
    context.SkipPastLikelyEnd(*context.position());
    return;
  } else if (kind != Lex::TokenKind::CloseCurlyBrace) {
    EmitUnexpectedTokenAndRecover(context);
    context.PushState(StateKind::MatchCaseLoopAfterDefault);
  }
}

auto HandleMatchCaseIntroducer(Context& context) -> void {
  auto state = context.PopState();

  context.AddLeafNode(NodeKind::MatchCaseIntroducer, context.Consume());
  context.PushState(state, StateKind::MatchCaseAfterPattern);
  context.PushState(StateKind::Pattern);
}

auto HandleMatchCaseAfterPattern(Context& context) -> void {
  auto state = context.PopState();
  if (state.has_error) {
    context.AddNode(NodeKind::MatchCaseStart, *context.position(),
                    /*has_error=*/true);
    context.AddNode(NodeKind::MatchCase, *context.position(),
                    /*has_error=*/true);
    context.SkipPastLikelyEnd(*context.position());
    return;
  }

  context.PushState(state, StateKind::MatchCaseStart);
  if (context.PositionIs(Lex::TokenKind::If)) {
    context.PushState(StateKind::MatchCaseGuardFinish);
    context.AddLeafNode(NodeKind::MatchCaseGuardIntroducer, context.Consume());
    auto open_paren = context.ConsumeIf(Lex::TokenKind::OpenParen);
    if (open_paren) {
      context.AddLeafNode(NodeKind::MatchCaseGuardStart, *open_paren);
      context.PushState(StateKind::Expr);
    } else {
      if (!state.has_error) {
        CARBON_DIAGNOSTIC(ExpectedMatchCaseGuardOpenParen, Error,
                          "expected `(` after `if`");
        context.emitter().Emit(*context.position(),
                               ExpectedMatchCaseGuardOpenParen);
      }

      context.AddLeafNode(NodeKind::MatchCaseGuardStart, *context.position(),
                          /*has_error=*/true);
      context.AddInvalidParse(*context.position());
      state = context.PopState();
      context.AddNode(NodeKind::MatchCaseGuard, *context.position(),
                      /*has_error=*/true);
      state = context.PopState();
      context.AddNode(NodeKind::MatchCaseStart, *context.position(),
                      /*has_error=*/true);
      context.AddNode(NodeKind::MatchCase, *context.position(),
                      /*has_error=*/true);
      context.SkipPastLikelyEnd(*context.position());
      return;
    }
  }
}

auto HandleMatchCaseGuardFinish(Context& context) -> void {
  auto state = context.PopState();

  auto close_paren = context.ConsumeIf(Lex::TokenKind::CloseParen);
  if (close_paren) {
    context.AddNode(NodeKind::MatchCaseGuard, *close_paren, state.has_error);
  } else {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(ExpectedMatchCaseGuardCloseParen, Error,
                        "expected `)`");
      context.emitter().Emit(*context.position(),
                             ExpectedMatchCaseGuardCloseParen);
    }

    context.AddNode(NodeKind::MatchCaseGuard, *context.position(),
                    /*has_error=*/true);
    context.ReturnErrorOnState();
    context.SkipPastLikelyEnd(*context.position());
    return;
  }
}

auto HandleMatchCaseStart(Context& context) -> void {
  HandleStatementsBlockStart(context, StateKind::MatchCaseFinish,
                             NodeKind::MatchCaseEqualGreater,
                             NodeKind::MatchCaseStart, NodeKind::MatchCase);
}

auto HandleMatchCaseFinish(Context& context) -> void {
  auto state = context.PopState();
  context.AddNode(NodeKind::MatchCase,
                  context.ConsumeChecked(Lex::TokenKind::CloseCurlyBrace),
                  state.has_error);
}

auto HandleMatchDefaultIntroducer(Context& context) -> void {
  context.AddLeafNode(NodeKind::MatchDefaultIntroducer, context.Consume());

  HandleStatementsBlockStart(context, StateKind::MatchDefaultFinish,
                             NodeKind::MatchDefaultEqualGreater,
                             NodeKind::MatchDefaultStart,
                             NodeKind::MatchDefault);
}

auto HandleMatchDefaultFinish(Context& context) -> void {
  auto state = context.PopState();
  context.AddNode(NodeKind::MatchDefault,
                  context.ConsumeChecked(Lex::TokenKind::CloseCurlyBrace),
                  state.has_error);
}

auto HandleMatchStatementFinish(Context& context) -> void {
  auto state = context.PopState();
  context.AddNode(NodeKind::MatchStatement,
                  context.ConsumeChecked(Lex::TokenKind::CloseCurlyBrace),
                  state.has_error);
}

}  // namespace Carbon::Parse
