// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

auto HandleRequireAfterIntroducer(Context& context) -> void {
  auto state = context.PopState();
  state.kind = StateKind::RequireDecl;
  context.PushState(state);

  if (context.PositionIs(Lex::TokenKind::Impls)) {
    // impls <expression> ...
    context.AddLeafNode(NodeKind::RequireDefaultSelfImpls, context.Consume());
    context.PushState(StateKind::Expr);
  } else {
    // <expression> impls <expression>...
    context.PushState(StateKind::RequireBeforeImpls);
    context.PushStateForExpr(PrecedenceGroup::ForRequirements());
  }
}

auto HandleRequireBeforeImpls(Context& context) -> void {
  auto state = context.PopState();
  if (auto impls = context.ConsumeIf(Lex::TokenKind::Impls)) {
    context.AddNode(NodeKind::RequireTypeImpls, *impls, state.has_error);
    context.PushState(StateKind::Expr);
  } else {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(RequireExpectedImpls, Error,
                        "expected `impls` in `require` declaration");
      context.emitter().Emit(*context.position(), RequireExpectedImpls);
    }
    context.ReturnErrorOnState();
  }
}

auto HandleRequireDecl(Context& context) -> void {
  auto state = context.PopState();
  context.AddNodeExpectingDeclSemi(state, NodeKind::RequireDecl,
                                   Lex::TokenKind::Require,
                                   /*is_def_allowed=*/false);
}

}  // namespace Carbon::Parse
