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

  std::optional<TokenizedBuffer::Token> open_paren =
      context.ConsumeAndAddOpenParen(state.token, start_kind);
  if (open_paren) {
    state.token = *open_paren;
  }
  state.state = finish_state;
  context.PushState(state);

  if (!open_paren && context.PositionIs(TokenKind::OpenCurlyBrace)) {
    // For an open curly, assume the condition was completely omitted.
    // Expression parsing would treat the { as a struct, but instead assume it's
    // a code block and just emit an invalid parse.
    context.AddLeafNode(ParseNodeKind::InvalidParse, *context.position(),
                        /*has_error=*/true);
  } else {
    context.PushState(ParserState::Expression);
  }
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

  context.ConsumeAndAddCloseSymbol(state.token, state,
                                   ParseNodeKind::IfCondition);
}

auto ParserHandleParenConditionFinishAsWhile(ParserContext& context) -> void {
  auto state = context.PopState();

  context.ConsumeAndAddCloseSymbol(state.token, state,
                                   ParseNodeKind::WhileCondition);
}

}  // namespace Carbon
