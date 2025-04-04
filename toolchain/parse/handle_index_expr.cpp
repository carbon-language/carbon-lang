// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

auto HandleIndexExpr(Context& context) -> void {
  auto state = context.PopState();
  context.PushState(state, StateKind::IndexExprFinish);
  context.AddNode(NodeKind::IndexExprStart,
                  context.ConsumeChecked(Lex::TokenKind::OpenSquareBracket),
                  state.has_error);
  context.PushState(StateKind::Expr);
}

auto HandleIndexExprFinish(Context& context) -> void {
  auto state = context.PopState();

  context.ConsumeAndAddCloseSymbol(state.token, state, NodeKind::IndexExpr);
}

}  // namespace Carbon::Parse
