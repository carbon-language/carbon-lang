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

auto HandleTypeOfExpr(Context& context) -> void {
  auto state = context.PopState();
  auto keyword = context.ConsumeChecked(Lex::TokenKind::TypeOf);
  context.AddLeafNode(NodeKind::TypeOfExprKeyword, keyword);
  if (auto open_paren = context.ConsumeAndAddOpenParen(
          keyword, NodeKind::TypeOfExprOpenParen)) {
    // Stash the open paren token for use by ConsumeAndAddCloseSymbol.
    state.token = *open_paren;
  } else {
    state.has_error = true;
  }
  context.PushState(state, StateKind::TypeOfExprFinish);
  context.PushState(StateKind::Expr);
}

auto HandleTypeOfExprFinish(Context& context) -> void {
  auto state = context.PopState();
  context.ConsumeAndAddCloseSymbol(state, NodeKind::TypeOfExpr);
}

}  // namespace Carbon::Parse
