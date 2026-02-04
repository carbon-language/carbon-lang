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

auto HandleFormLiteral(Context& context) -> void {
  auto state = context.PopState();

  auto keyword = context.ConsumeChecked(Lex::TokenKind::Form);
  context.AddLeafNode(NodeKind::FormLiteralKeyword, keyword);
  if (auto paren = context.ConsumeAndAddOpenParen(
          keyword, NodeKind::FormLiteralOpenParen)) {
    // Stash the open paren token for use by ConsumeAndAddCloseSymbol.
    state.token = *paren;
  } else {
    state.has_error = true;
  }
  if (!context.ConsumeAndAddLeafNodeIf(Lex::TokenKind::Ref,
                                       NodeKind::RefCategoryModifier) &&
      !context.ConsumeAndAddLeafNodeIf(Lex::TokenKind::Var,
                                       NodeKind::VarCategoryModifier) &&
      !context.ConsumeAndAddLeafNodeIf(Lex::TokenKind::Val,
                                       NodeKind::ValCategoryModifier)) {
    // If we didn't even have an open paren, diagnosing the lack of a category
    // probably won't be useful.
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(ExpectedCategoryModifier, Error,
                        "expected `ref`, `var`, or `val` after `form(`");
      context.emitter().Emit(*context.position(), ExpectedCategoryModifier);
      state.has_error = true;
    }
    context.AddInvalidParse(*context.position());
  }
  context.PushState(state, StateKind::FormLiteralFinish);
  context.PushState(StateKind::Expr);
}

auto HandleFormLiteralFinish(Context& context) -> void {
  auto state = context.PopState();
  context.ConsumeAndAddCloseSymbol(state.token, state, NodeKind::FormLiteral);
}

}  // namespace Carbon::Parse
