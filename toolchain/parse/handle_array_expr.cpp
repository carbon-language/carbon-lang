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
  auto array_token = context.ConsumeChecked(Lex::TokenKind::Array);
  context.AddLeafNode(NodeKind::ArrayExprKeyword, array_token, state.has_error);
  if (auto open_paren = context.ConsumeAndAddOpenParen(
          array_token, NodeKind::ArrayExprOpenParen)) {
    state.token = *open_paren;
  } else {
    state.has_error = true;
  }
  context.PushState(state, State::ArrayExprComma);
  context.PushState(State::Expr);
}

auto HandleArrayExprComma(Context& context) -> void {
  auto state = context.PopState();
  if (!context.ConsumeAndAddLeafNodeIf(Lex::TokenKind::Comma,
                                       NodeKind::ArrayExprComma)) {
    context.AddLeafNode(NodeKind::ArrayExprComma, *context.position(), true);
    CARBON_DIAGNOSTIC(ExpectedArrayComma, Error,
                      "expected `,` in `array(Type, Count)`");
    context.emitter().Emit(*context.position(), ExpectedArrayComma);
    state.has_error = true;
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
