// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/token_kind.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/parse/state.h"

namespace Carbon::Parse {

auto HandleRequire(Context& context) -> void {
  context.PopState();

  context.PushState(State::RequireFinish);
  context.PushState(State::Impls);
  context.PushState(State::Expr);
}

auto HandleImpls(Context& context) -> void {
  auto state = context.PopState();

  if (auto impls = context.ConsumeIf(Lex::TokenKind::Impls)) {
    // context.AddNode(NodeKind::ImplsTypeAs, *impls, state.has_error);
    state.token = *impls;
    state.state = State::ImplsFinish;
    context.PushState(state);
    context.PushState(State::Expr);
  } else {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(RequireExpectedImpls, Error,
                        "Expected `impls` in `require` declaration.");
      context.emitter().Emit(*context.position(), RequireExpectedImpls);
    }
    context.ReturnErrorOnState();
  }
}

auto HandleImplsFinish(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::Impls, state.token, state.has_error);
}

auto HandleRequireFinish(Context& context) -> void {
  auto state = context.PopState();

  context.AddNodeExpectingDeclSemi(state, NodeKind::Require,
                                   Lex::TokenKind::Require,
                                   /*is_def_allowed=*/false);
}

}  // namespace Carbon::Parse
