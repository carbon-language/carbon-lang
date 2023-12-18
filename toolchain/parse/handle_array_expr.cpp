// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/token_kind.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/parse/state.h"

namespace Carbon::Parse {

auto HandleArrayExpr(Context& context) -> void {
  auto state = context.PopState();
  context.AddLeafNode(NodeKind::ArrayExprStart,
                      context.ConsumeChecked(Lex::TokenKind::OpenSquareBracket),
                      state.has_error);
  context.PushState(state, State::ArrayExprSemi);
  context.PushState(State::Expr);
}

auto HandleArrayExprSemi(Context& context) -> void {
  auto state = context.PopState();
  auto semi = context.ConsumeIf(Lex::TokenKind::Semi);
  if (!semi) {
    context.AddNode(NodeKind::ArrayExprSemi, *context.position(),
                    state.subtree_start, true);
    CARBON_DIAGNOSTIC(ExpectedArraySemi, Error, "Expected `;` in array type.");
    context.emitter().Emit(*context.position(), ExpectedArraySemi);
    state.has_error = true;
  } else {
    context.AddNode(NodeKind::ArrayExprSemi, *semi, state.subtree_start,
                    state.has_error);
  }
  context.PushState(state, State::ArrayExprFinish);
  if (!context.PositionIs(Lex::TokenKind::CloseSquareBracket)) {
    context.PushState(State::Expr);
  }
}

auto HandleArrayExprFinish(Context& context) -> void {
  auto state = context.PopState();
  context.ConsumeAndAddCloseSymbol(*(Lex::TokenIterator(state.token)), state,
                                   NodeKind::ArrayExpr);
}

}  // namespace Carbon::Parse
