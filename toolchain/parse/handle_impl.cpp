// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

static auto ExpectAsOrTypeExpression(Context& context) -> void {
  if (context.PositionIs(Lex::TokenKind::As)) {
    // as <expression> ...
    context.AddLeafNode(NodeKind::DefaultSelfImplAs, context.Consume());
    context.PushState(StateKind::Expr);
  } else {
    // <expression> as <expression>...
    context.PushState(StateKind::ImplBeforeAs);
    context.PushStateForExpr(PrecedenceGroup::ForImplAs());
  }
}

auto HandleImplAfterIntroducer(Context& context) -> void {
  auto state = context.PopState();
  state.kind = StateKind::DeclOrDefinitionAsImpl;
  context.PushState(state);

  if (context.PositionIs(Lex::TokenKind::Forall)) {
    // forall [<implicit parameter list>] ...
    context.PushState(StateKind::ImplAfterForall);
    context.AddLeafNode(NodeKind::Forall, context.Consume());
    if (context.PositionIs(Lex::TokenKind::OpenSquareBracket)) {
      context.PushState(StateKind::PatternListAsImplicit);
    } else {
      CARBON_DIAGNOSTIC(ImplExpectedAfterForall, Error,
                        "expected `[` after `forall` in `impl` declaration");
      context.emitter().Emit(*context.position(), ImplExpectedAfterForall);
      context.ReturnErrorOnState();
      // If we aren't producing a node from the PatternListAsImplicit state,
      // we still need to create a node to be the child of the `ImplForall`
      // token created in the `ImplAfterForall` state.
      context.AddInvalidParse(*context.position());
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
  // One of:
  //   as <expression> ...
  //   <expression> as <expression>...
  ExpectAsOrTypeExpression(context);
}

auto HandleImplBeforeAs(Context& context) -> void {
  auto state = context.PopState();
  if (auto as = context.ConsumeIf(Lex::TokenKind::As)) {
    context.AddNode(NodeKind::TypeImplAs, *as, state.has_error);
    context.PushState(StateKind::Expr);
  } else {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(ImplExpectedAs, Error,
                        "expected `as` in `impl` declaration");
      context.emitter().Emit(*context.position(), ImplExpectedAs);
    }
    context.ReturnErrorOnState();
  }
}

}  // namespace Carbon::Parse
