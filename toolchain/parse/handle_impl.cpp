// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

static auto ExpectAsOrTypeExpression(Carbon::Parse::Context& context) -> void {
  if (context.PositionIs(Lex::TokenKind::As)) {
    // as <expression> ...
    context.AddLeafNode(NodeKind::ImplAs, context.Consume());
    context.PushState(State::Expr);
  } else {
    // <expression> as <expression>...
    context.PushState(State::ImplBeforeAs);
    context.PushStateForExpr(PrecedenceGroup::ForImplAs());
  }
}

auto HandleImplAfterIntroducer(Carbon::Parse::Context& context) -> void {
  auto state = context.PopState();
  state.state = State::DeclOrDefinitionAsImpl;
  context.PushState(state);

  if (context.PositionIs(Lex::TokenKind::Forall)) {
    // forall [<implicit parameter list>] ...
    context.PushState(State::ImplAfterForall);
    context.ConsumeAndDiscard();
    if (context.PositionIs(Lex::TokenKind::OpenSquareBracket)) {
      context.PushState(State::PatternListAsImplicit);
    } else {
      CARBON_DIAGNOSTIC(ImplExpectedAfterForall, Error,
                        "Expected `[` after `forall` in `impl` declaration.");
      context.emitter().Emit(*context.position(), ImplExpectedAfterForall);
      context.ReturnErrorOnState();
      // If we aren't producing a node from the PatternListAsImplicit state,
      // we still need to create a node to be the child of the `ImplForall`
      // token created in the `ImplAfterForall` state.
      context.AddLeafNode(NodeKind::InvalidParse, *context.position(),
                          /*has_error=*/true);
    }
  } else {
    // One of:
    //   as <expression> ...
    //   <expression> as <expression>...
    ExpectAsOrTypeExpression(context);
  }
}

auto HandleImplAfterForall(Carbon::Parse::Context& context) -> void {
  auto state = context.PopState();
  if (state.has_error) {
    context.ReturnErrorOnState();
  }
  context.AddNode(NodeKind::ImplForall, state.token, state.subtree_start,
                  state.has_error);

  // One of:
  //   as <expression> ...
  //   <expression> as <expression>...
  ExpectAsOrTypeExpression(context);
}

auto HandleImplBeforeAs(Carbon::Parse::Context& context) -> void {
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
