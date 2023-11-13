// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

// Handles ParamAs(Implicit|Regular).
static auto HandleParam(Context& context, State pattern_state,
                        State finish_state) -> void {
  context.PopAndDiscardState();

  context.PushState(finish_state);
  context.PushState(pattern_state);
}

auto HandleParamAsImplicit(Context& context) -> void {
  HandleParam(context, State::PatternAsImplicitParam,
              State::ParamFinishAsImplicit);
}

auto HandleParamAsRegular(Context& context) -> void {
  HandleParam(context, State::PatternAsParam, State::ParamFinishAsRegular);
}

// Handles ParamFinishAs(Implicit|Regular).
static auto HandleParamFinish(Context& context, Lex::TokenKind close_token,
                              State param_state) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    context.ReturnErrorOnState();
  }

  if (context.ConsumeListToken(NodeKind::ParamListComma, close_token,
                               state.has_error) ==
      Context::ListTokenKind::Comma) {
    context.PushState(param_state);
  }
}

auto HandleParamFinishAsImplicit(Context& context) -> void {
  HandleParamFinish(context, Lex::TokenKind::CloseSquareBracket,
                    State::ParamAsImplicit);
}

auto HandleParamFinishAsRegular(Context& context) -> void {
  HandleParamFinish(context, Lex::TokenKind::CloseParen, State::ParamAsRegular);
}

// Handles ParamListAs(Implicit|Regular).
static auto HandleParamList(Context& context, NodeKind parse_node_kind,
                            Lex::TokenKind open_token_kind,
                            Lex::TokenKind close_token_kind, State param_state,
                            State finish_state) -> void {
  context.PopAndDiscardState();

  context.PushState(finish_state);
  context.AddLeafNode(parse_node_kind, context.ConsumeChecked(open_token_kind));

  if (!context.PositionIs(close_token_kind)) {
    context.PushState(param_state);
  }
}

auto HandleParamListAsImplicit(Context& context) -> void {
  HandleParamList(context, NodeKind::ImplicitParamListStart,
                  Lex::TokenKind::OpenSquareBracket,
                  Lex::TokenKind::CloseSquareBracket, State::ParamAsImplicit,
                  State::ParamListFinishAsImplicit);
}

auto HandleParamListAsRegular(Context& context) -> void {
  HandleParamList(context, NodeKind::ParamListStart, Lex::TokenKind::OpenParen,
                  Lex::TokenKind::CloseParen, State::ParamAsRegular,
                  State::ParamListFinishAsRegular);
}

// Handles ParamListFinishAs(Implicit|Regular).
static auto HandleParamListFinish(Context& context, NodeKind parse_node_kind,
                                  Lex::TokenKind token_kind) -> void {
  auto state = context.PopState();

  context.AddNode(parse_node_kind, context.ConsumeChecked(token_kind),
                  state.subtree_start, state.has_error);
}

auto HandleParamListFinishAsImplicit(Context& context) -> void {
  HandleParamListFinish(context, NodeKind::ImplicitParamList,
                        Lex::TokenKind::CloseSquareBracket);
}

auto HandleParamListFinishAsRegular(Context& context) -> void {
  HandleParamListFinish(context, NodeKind::ParamList,
                        Lex::TokenKind::CloseParen);
}

}  // namespace Carbon::Parse
