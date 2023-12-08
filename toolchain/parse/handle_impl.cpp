// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

auto HandleImplAfterIntroducer(Carbon::Parse::Context& context) -> void {
  auto state = context.PopState();
  state.state = State::DeclOrDefinitionAsImpl;
  context.PushState(state);

  switch (context.PositionKind()) {
    case Lex::TokenKind::Forall:
      context.PushState(State::ImplAfterForall);
      context.ConsumeAndDiscard();
      if (context.PositionIs(Lex::TokenKind::OpenSquareBracket)) {
        context.PushState(State::ParamListAsImplicit);
      } else {
        CARBON_DIAGNOSTIC(ImplExpectedAfterForall, Error,
                          "Expected `[` after `forall` in `impl` declaration.");
        context.emitter().Emit(*context.position(), ImplExpectedAfterForall);
        context.ReturnErrorOnState();
      }
      break;

    case Lex::TokenKind::As:
      context.AddLeafNode(NodeKind::ImplAs, context.Consume());
      context.PushState(State::Expr);
      break;

    default:
      context.PushState(State::ImplAs);
      context.PushStateForExpr(PrecedenceGroup::ForImplAs());
      break;
  }
}

auto HandleImplAfterForall(Carbon::Parse::Context& context) -> void {
  auto state = context.PopState();
  if (state.has_error) {
    context.ReturnErrorOnState();
    return;
  }

  context.AddNode(NodeKind::ImplForall, state.token, state.subtree_start,
                  state.has_error);

  if (context.PositionKind() == Lex::TokenKind::As) {
    context.AddLeafNode(NodeKind::ImplAs, context.Consume());
    context.PushState(State::Expr);
  } else {
    context.PushState(State::ImplAs);
    context.PushStateForExpr(PrecedenceGroup::ForImplAs());
  }
}

auto HandleImplAs(Carbon::Parse::Context& context) -> void {
  auto state = context.PopState();
  if (state.has_error) {
    context.ReturnErrorOnState();
    return;
  }
  if (auto as = context.ConsumeIf(Lex::TokenKind::As)) {
    context.AddLeafNode(NodeKind::ImplAs, *as);
    context.PushState(State::Expr);
  } else {
    CARBON_DIAGNOSTIC(ImplExpectedAs, Error,
                      "Expected `as` in `impl` declaration.");
    context.emitter().Emit(*context.position(), ImplExpectedAs);
    context.ReturnErrorOnState();
  }
}

}  // namespace Carbon::Parse
