// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

// Handles PatternListElementAs(Implicit|Tuple).
static auto HandlePatternListElement(Context& context, State pattern_state,
                                     State finish_state) -> void {
  context.PopAndDiscardState();

  context.PushState(finish_state);
  context.PushState(pattern_state);
}

auto HandlePatternListElementAsImplicit(Context& context) -> void {
  HandlePatternListElement(context, State::BindingPattern,
                           State::PatternListElementFinishAsImplicit);
}

auto HandlePatternListElementAsTuple(Context& context) -> void {
  HandlePatternListElement(context, State::BindingPattern,
                           State::PatternListElementFinishAsTuple);
}

// Handles PatternListElementFinishAs(Implicit|Tuple).
static auto HandlePatternListElementFinish(Context& context,
                                           Lex::TokenKind close_token,
                                           State param_state) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    context.ReturnErrorOnState();
  }

  if (context.ConsumeListToken(NodeKind::PatternListComma, close_token,
                               state.has_error) ==
      Context::ListTokenKind::Comma) {
    context.PushState(param_state);
  }
}

auto HandlePatternListElementFinishAsImplicit(Context& context) -> void {
  HandlePatternListElementFinish(context, Lex::TokenKind::CloseSquareBracket,
                                 State::PatternListElementAsImplicit);
}

auto HandlePatternListElementFinishAsTuple(Context& context) -> void {
  HandlePatternListElementFinish(context, Lex::TokenKind::CloseParen,
                                 State::PatternListElementAsTuple);
}

// Handles PatternListAs(Implicit|Tuple).
static auto HandlePatternList(Context& context, NodeKind node_kind,
                              Lex::TokenKind open_token_kind,
                              Lex::TokenKind close_token_kind,
                              State param_state, State finish_state) -> void {
  context.PopAndDiscardState();

  context.PushState(finish_state);
  context.AddLeafNode(node_kind, context.ConsumeChecked(open_token_kind));

  if (!context.PositionIs(close_token_kind)) {
    context.PushState(param_state);
  }
}

auto HandlePatternListAsImplicit(Context& context) -> void {
  HandlePatternList(
      context, NodeKind::ImplicitParamListStart,
      Lex::TokenKind::OpenSquareBracket, Lex::TokenKind::CloseSquareBracket,
      State::PatternListElementAsImplicit, State::PatternListFinishAsImplicit);
}

auto HandlePatternListAsTuple(Context& context) -> void {
  HandlePatternList(context, NodeKind::TuplePatternStart,
                    Lex::TokenKind::OpenParen, Lex::TokenKind::CloseParen,
                    State::PatternListElementAsTuple,
                    State::PatternListFinishAsTuple);
}

// Handles PatternListFinishAs(Implicit|Tuple).
static auto HandlePatternListFinish(Context& context, NodeKind node_kind,
                                    Lex::TokenKind token_kind) -> void {
  auto state = context.PopState();

  context.AddNode(node_kind, context.ConsumeChecked(token_kind),
                  state.subtree_start, state.has_error);
}

auto HandlePatternListFinishAsImplicit(Context& context) -> void {
  HandlePatternListFinish(context, NodeKind::ImplicitParamList,
                          Lex::TokenKind::CloseSquareBracket);
}

auto HandlePatternListFinishAsTuple(Context& context) -> void {
  HandlePatternListFinish(context, NodeKind::TuplePattern,
                          Lex::TokenKind::CloseParen);
}

}  // namespace Carbon::Parse
