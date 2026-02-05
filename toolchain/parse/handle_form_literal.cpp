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
  context.PushState(state, StateKind::FormLiteralFinish);
  if (state.has_error) {
    context.AddInvalidParse(keyword);
  } else {
    context.PushState(StateKind::PrimitiveForm, keyword);
  }
}

auto HandlePrimitiveForm(Context& context) -> void {
  auto state = context.PopState();
  if (context.PositionIs(Lex::TokenKind::Ref) ||
      context.PositionIs(Lex::TokenKind::Var) ||
      context.PositionIs(Lex::TokenKind::Val)) {
    state.token = context.Consume();
  } else {
    CARBON_DIAGNOSTIC(ExpectedCategoryModifier, Error,
                      "expected `ref`, `var`, or `val` after `form(`");
    context.emitter().Emit(*context.position(), ExpectedCategoryModifier);
    state.has_error = true;
  }
  context.PushState(state, StateKind::PrimitiveFormFinish);
  context.PushState(StateKind::Expr);
}

auto HandlePrimitiveFormFinish(Context& context) -> void {
  auto state = context.PopState();

  // Arbitrary default, only used in error recovery.
  auto node_kind = NodeKind::ValPrimitiveForm;
  switch (context.tokens().GetKind(state.token)) {
    case Lex::TokenKind::Ref:
      node_kind = NodeKind::RefPrimitiveForm;
      break;
    case Lex::TokenKind::Var:
      node_kind = NodeKind::VarPrimitiveForm;
      break;
    case Lex::TokenKind::Val:
      node_kind = NodeKind::ValPrimitiveForm;
      break;
    default:
      CARBON_CHECK(state.has_error);
      // Use the default node_kind set earlier for error recovery.
      break;
  }
  context.AddNode(node_kind, state.token, state.has_error);
}

auto HandleFormLiteralFinish(Context& context) -> void {
  auto state = context.PopState();
  context.ConsumeAndAddCloseSymbol(state.token, state, NodeKind::FormLiteral);
}

}  // namespace Carbon::Parse
