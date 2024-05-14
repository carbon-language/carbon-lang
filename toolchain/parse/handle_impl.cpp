// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

static auto ExpectAsOrTypeExpression(Context& context) -> void {
  if (context.PositionIs(Lex::TokenKind::As)) {
    // as <expression> ...
    context.AddLeafNode(NodeKind::DefaultSelfImplAs, context.Consume());
    context.PushState(State::Expr);
  } else {
    // <expression> as <expression>...
    context.PushState(State::ImplBeforeAs);
    context.PushStateForExpr(PrecedenceGroup::ForImplAs());
  }
}

auto HandleImplAfterIntroducer(Context& context) -> void {
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

auto HandleImplAfterForall(Context& context) -> void {
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

auto HandleImplBeforeAs(Context& context) -> void {
  auto state = context.PopState();
  if (auto as = context.ConsumeIf(Lex::TokenKind::As)) {
    context.AddNode(NodeKind::TypeImplAs, *as, state.subtree_start,
                    state.has_error);
    context.PushState(State::Expr);
  } else {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(ImplExpectedAs, Error,
                        "Expected `as` in `impl` declaration.");
      context.emitter().Emit(*context.position(), ImplExpectedAs);
    }
    context.ReturnErrorOnState();
  }
}

}  // namespace Carbon::Parse
