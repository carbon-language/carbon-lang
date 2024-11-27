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

auto HandleArrayExpr(Context& context) -> void {
  auto state = context.PopState();
  context.AddLeafNode(NodeKind::ArrayExprKeyword,
                      context.ConsumeChecked(Lex::TokenKind::Array),
                      state.has_error);
  if (auto open_paren = context.ConsumeIf(Lex::TokenKind::OpenParen)) {
    context.AddNode(NodeKind::ArrayExprStart, *open_paren, state.has_error);
    state.token = *open_paren;
  } else {
    context.AddNode(NodeKind::ArrayExprStart, *context.position(), true);
    CARBON_DIAGNOSTIC(ExpectedArrayParen, Error, "expected `(` after `array`");
    context.emitter().Emit(*context.position(), ExpectedArrayParen);
    state.has_error = true;
  }
  context.PushState(state, State::ArrayExprComma);
  context.PushState(State::Expr);
}

auto HandleArrayExprComma(Context& context) -> void {
  auto state = context.PopState();
  auto comma = context.ConsumeIf(Lex::TokenKind::Comma);
  if (!comma) {
    context.AddLeafNode(NodeKind::ArrayExprComma, *context.position(), true);
    CARBON_DIAGNOSTIC(ExpectedArrayComma, Error,
                      "expected `,` in array(Type, Count)");
    context.emitter().Emit(*context.position(), ExpectedArrayComma);
    state.has_error = true;
  } else {
    context.AddLeafNode(NodeKind::ArrayExprComma, *comma, state.has_error);
  }
  context.PushState(state, State::ArrayExprFinish);
  context.PushState(State::Expr);
}

auto HandleArrayExprFinish(Context& context) -> void {
  auto state = context.PopState();
  context.ConsumeAndAddCloseSymbol(*(Lex::TokenIterator(state.token)), state,
                                   NodeKind::ArrayExpr);
}

}  // namespace Carbon::Parse
