// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

// Handles PatternListElementAs(Tuple|Explicit|Implicit).
static auto HandlePatternListElement(Context& context, StateKind pattern_state,
                                     StateKind finish_state_kind) -> void {
  auto state = context.PopState();

  context.PushStateForPattern(finish_state_kind, state.in_var_pattern);
  context.PushStateForPattern(pattern_state, state.in_var_pattern);
}

auto HandlePatternListElementAsTuple(Context& context) -> void {
  HandlePatternListElement(context, StateKind::Pattern,
                           StateKind::PatternListElementFinishAsTuple);
}

auto HandlePatternListElementAsExplicit(Context& context) -> void {
  HandlePatternListElement(context, StateKind::Pattern,
                           StateKind::PatternListElementFinishAsExplicit);
}

auto HandlePatternListElementAsImplicit(Context& context) -> void {
  HandlePatternListElement(context, StateKind::Pattern,
                           StateKind::PatternListElementFinishAsImplicit);
}

// Handles PatternListElementFinishAs(Tuple|Explicit|Implicit).
static auto HandlePatternListElementFinish(Context& context,
                                           Lex::TokenKind close_token,
                                           StateKind param_state_kind) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    context.ReturnErrorOnState();
  }

  if (context.ConsumeListToken(NodeKind::PatternListComma, close_token,
                               state.has_error) ==
      Context::ListTokenKind::Comma) {
    context.PushStateForPattern(param_state_kind, state.in_var_pattern);
  }
}

auto HandlePatternListElementFinishAsTuple(Context& context) -> void {
  HandlePatternListElementFinish(context, Lex::TokenKind::CloseParen,
                                 StateKind::PatternListElementAsTuple);
}

auto HandlePatternListElementFinishAsExplicit(Context& context) -> void {
  HandlePatternListElementFinish(context, Lex::TokenKind::CloseParen,
                                 StateKind::PatternListElementAsExplicit);
}

auto HandlePatternListElementFinishAsImplicit(Context& context) -> void {
  HandlePatternListElementFinish(context, Lex::TokenKind::CloseSquareBracket,
                                 StateKind::PatternListElementAsImplicit);
}

// Handles PatternListAs(Tuple|Explicit|Implicit).
static auto HandlePatternList(Context& context, NodeKind node_kind,
                              Lex::TokenKind open_token_kind,
                              Lex::TokenKind close_token_kind,
                              StateKind param_state, StateKind finish_state)
    -> void {
  auto state = context.PopState();

  context.PushStateForPattern(finish_state, state.in_var_pattern);
  context.AddLeafNode(node_kind, context.ConsumeChecked(open_token_kind));

  if (!context.PositionIs(close_token_kind)) {
    context.PushStateForPattern(param_state, state.in_var_pattern);
  }
}

auto HandlePatternListAsTuple(Context& context) -> void {
  HandlePatternList(context, NodeKind::TuplePatternStart,
                    Lex::TokenKind::OpenParen, Lex::TokenKind::CloseParen,
                    StateKind::PatternListElementAsTuple,
                    StateKind::PatternListFinishAsTuple);
}

auto HandlePatternListAsExplicit(Context& context) -> void {
  HandlePatternList(context, NodeKind::ExplicitParamListStart,
                    Lex::TokenKind::OpenParen, Lex::TokenKind::CloseParen,
                    StateKind::PatternListElementAsExplicit,
                    StateKind::PatternListFinishAsExplicit);
}

auto HandlePatternListAsImplicit(Context& context) -> void {
  HandlePatternList(context, NodeKind::ImplicitParamListStart,
                    Lex::TokenKind::OpenSquareBracket,
                    Lex::TokenKind::CloseSquareBracket,
                    StateKind::PatternListElementAsImplicit,
                    StateKind::PatternListFinishAsImplicit);
}

// Handles PatternListFinishAs(Tuple|Explicit|Implicit).
static auto HandlePatternListFinish(Context& context, NodeKind node_kind,
                                    Lex::TokenKind token_kind) -> void {
  auto state = context.PopState();

  context.AddNode(node_kind, context.ConsumeChecked(token_kind),
                  state.has_error);
}

auto HandlePatternListFinishAsTuple(Context& context) -> void {
  HandlePatternListFinish(context, NodeKind::TuplePattern,
                          Lex::TokenKind::CloseParen);
}

auto HandlePatternListFinishAsExplicit(Context& context) -> void {
  HandlePatternListFinish(context, NodeKind::ExplicitParamList,
                          Lex::TokenKind::CloseParen);
}

auto HandlePatternListFinishAsImplicit(Context& context) -> void {
  HandlePatternListFinish(context, NodeKind::ImplicitParamList,
                          Lex::TokenKind::CloseSquareBracket);
}

}  // namespace Carbon::Parse
