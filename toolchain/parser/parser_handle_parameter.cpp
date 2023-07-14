// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_context.h"

namespace Carbon {

// Handles ParameterAs(Deduced|Regular).
static auto ParserHandleParameter(ParserContext& context,
                                  ParserState pattern_state,
                                  ParserState finish_state) -> void {
  context.PopAndDiscardState();

  context.PushState(finish_state);
  context.PushState(pattern_state);
}

auto ParserHandleParameterAsDeduced(ParserContext& context) -> void {
  ParserHandleParameter(context, ParserState::PatternAsDeducedParameter,
                        ParserState::ParameterFinishAsDeduced);
}

auto ParserHandleParameterAsRegular(ParserContext& context) -> void {
  ParserHandleParameter(context, ParserState::PatternAsParameter,
                        ParserState::ParameterFinishAsRegular);
}

// Handles ParameterFinishAs(Deduced|Regular).
static auto ParserHandleParameterFinish(ParserContext& context,
                                        TokenKind close_token,
                                        ParserState param_state) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    context.ReturnErrorOnState();
  }

  if (context.ConsumeListToken(ParseNodeKind::ParameterListComma, close_token,
                               state.has_error) ==
      ParserContext::ListTokenKind::Comma) {
    context.PushState(param_state);
  }
}

auto ParserHandleParameterFinishAsDeduced(ParserContext& context) -> void {
  ParserHandleParameterFinish(context, TokenKind::CloseSquareBracket,
                              ParserState::ParameterAsDeduced);
}

auto ParserHandleParameterFinishAsRegular(ParserContext& context) -> void {
  ParserHandleParameterFinish(context, TokenKind::CloseParen,
                              ParserState::ParameterAsRegular);
}

// Handles ParameterListAs(Deduced|Regular).
static auto ParserHandleParameterList(ParserContext& context,
                                      ParseNodeKind parse_node_kind,
                                      TokenKind open_token_kind,
                                      TokenKind close_token_kind,
                                      ParserState param_state,
                                      ParserState finish_state) -> void {
  context.PopAndDiscardState();

  context.PushState(finish_state);
  context.AddLeafNode(parse_node_kind, context.ConsumeChecked(open_token_kind));

  if (!context.PositionIs(close_token_kind)) {
    context.PushState(param_state);
  }
}

auto ParserHandleParameterListAsDeduced(ParserContext& context) -> void {
  ParserHandleParameterList(context, ParseNodeKind::DeducedParameterListStart,
                            TokenKind::OpenSquareBracket,
                            TokenKind::CloseSquareBracket,
                            ParserState::ParameterAsDeduced,
                            ParserState::ParameterListFinishAsDeduced);
}

auto ParserHandleParameterListAsRegular(ParserContext& context) -> void {
  ParserHandleParameterList(context, ParseNodeKind::ParameterListStart,
                            TokenKind::OpenParen, TokenKind::CloseParen,
                            ParserState::ParameterAsRegular,
                            ParserState::ParameterListFinishAsRegular);
}

// Handles ParameterListFinishAs(Deduced|Regular).
static auto ParserHandleParameterListFinish(ParserContext& context,
                                            ParseNodeKind parse_node_kind,
                                            TokenKind token_kind) -> void {
  auto state = context.PopState();

  context.AddNode(parse_node_kind, context.ConsumeChecked(token_kind),
                  state.subtree_start, state.has_error);
}

auto ParserHandleParameterListFinishAsDeduced(ParserContext& context) -> void {
  ParserHandleParameterListFinish(context, ParseNodeKind::DeducedParameterList,
                                  TokenKind::CloseSquareBracket);
}

auto ParserHandleParameterListFinishAsRegular(ParserContext& context) -> void {
  ParserHandleParameterListFinish(context, ParseNodeKind::ParameterList,
                                  TokenKind::CloseParen);
}

}  // namespace Carbon
