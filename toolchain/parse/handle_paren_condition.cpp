// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

// Handles ParenConditionAs(If|While|Match).
static auto HandleParenCondition(Context& context, NodeKind start_kind,
                                 StateKind finish_state_kind) -> void {
  auto state = context.PopState();

  std::optional<Lex::TokenIndex> open_paren =
      context.ConsumeAndAddOpenParen(state.token, start_kind);
  if (open_paren) {
    state.token = *open_paren;
  }
  context.PushState(state, finish_state_kind);

  if (!open_paren && context.PositionIs(Lex::TokenKind::OpenCurlyBrace)) {
    // For an open curly, assume the condition was completely omitted.
    // Expression parsing would treat the { as a struct, but instead assume it's
    // a code block and just emit an invalid parse.
    context.AddInvalidParse(*context.position());
  } else {
    context.PushState(StateKind::Expr);
  }
}

auto HandleParenConditionAsIf(Context& context) -> void {
  HandleParenCondition(context, NodeKind::IfConditionStart,
                       StateKind::ParenConditionFinishAsIf);
}

auto HandleParenConditionAsWhile(Context& context) -> void {
  HandleParenCondition(context, NodeKind::WhileConditionStart,
                       StateKind::ParenConditionFinishAsWhile);
}

auto HandleParenConditionAsMatch(Context& context) -> void {
  HandleParenCondition(context, NodeKind::MatchConditionStart,
                       StateKind::ParenConditionFinishAsMatch);
}

auto HandleParenConditionFinishAsIf(Context& context) -> void {
  auto state = context.PopState();

  context.ConsumeAndAddCloseSymbol(state.token, state, NodeKind::IfCondition);
}

auto HandleParenConditionFinishAsWhile(Context& context) -> void {
  auto state = context.PopState();

  context.ConsumeAndAddCloseSymbol(state.token, state,
                                   NodeKind::WhileCondition);
}

auto HandleParenConditionFinishAsMatch(Context& context) -> void {
  auto state = context.PopState();

  context.ConsumeAndAddCloseSymbol(state.token, state,
                                   NodeKind::MatchCondition);
}

}  // namespace Carbon::Parse
