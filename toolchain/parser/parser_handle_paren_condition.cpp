// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_context.h"

namespace Carbon {

// Handles ParenConditionAs(If|While).
static auto ParserHandleParenCondition(ParserContext& context,
                                       ParseNodeKind start_kind,
                                       ParserState finish_state) -> void {
  auto state = context.PopState();

  context.ConsumeAndAddOpenParen(state.token, start_kind);

  state.state = finish_state;
  context.PushState(state);
  context.PushState(ParserState::Expression);
}

auto ParserHandleParenConditionAsIf(ParserContext& context) -> void {
  ParserHandleParenCondition(context, ParseNodeKind::IfConditionStart,
                             ParserState::ParenConditionFinishAsIf);
}

auto ParserHandleParenConditionAsWhile(ParserContext& context) -> void {
  ParserHandleParenCondition(context, ParseNodeKind::WhileConditionStart,
                             ParserState::ParenConditionFinishAsWhile);
}

auto ParserHandleParenConditionFinishAsIf(ParserContext& context) -> void {
  auto state = context.PopState();

  context.ConsumeAndAddCloseParen(state, ParseNodeKind::IfCondition);
}

auto ParserHandleParenConditionFinishAsWhile(ParserContext& context) -> void {
  auto state = context.PopState();

  context.ConsumeAndAddCloseParen(state, ParseNodeKind::WhileCondition);
}

}  // namespace Carbon
