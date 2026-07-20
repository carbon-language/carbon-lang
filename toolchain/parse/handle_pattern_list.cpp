// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

// Handles PatternListElementAs(Tuple|Explicit|Implicit).
static auto HandlePatternListElement(Context& context, StateKind pattern_state,
                                     StateKind finish_state_kind) -> void {
  auto state = context.PopState();

  context.PushStateForPattern(finish_state_kind, state.in_var_pattern,
                              state.in_unused_pattern, state.in_struct_pattern,
                              state.binding_context, state.ambient_precedence);
  context.PushStateForPattern(pattern_state, state.in_var_pattern,
                              state.in_unused_pattern, state.in_struct_pattern,
                              state.binding_context, state.ambient_precedence);
}

auto HandlePatternListElementAsTuple(Context& context) -> void {
  HandlePatternListElement(context, StateKind::Pattern,
                           StateKind::PatternListElementFinishAsTuple);
}

auto HandlePatternListElementAsStruct(Context& context) -> void {
  switch (context.PositionKind()) {
    case Lex::TokenKind::Period:
      HandlePatternListElement(context,
                               StateKind::StructPatternFieldAfterDesignator,
                               StateKind::PatternListElementFinishAsStruct);
      context.PushState(StateKind::PeriodAsStruct);
      break;
    case Lex::TokenKind::Underscore:
      HandlePatternListElement(context, StateKind::StructPatternUnderscore,
                               StateKind::PatternListElementFinishAsStruct);
      break;
    default:
      HandlePatternListElement(context, StateKind::Pattern,
                               StateKind::PatternListElementFinishAsStruct);
      break;
  }
}

auto HandlePatternListElementAsExplicit(Context& context) -> void {
  HandlePatternListElement(context, StateKind::Pattern,
                           StateKind::PatternListElementFinishAsExplicit);
}

auto HandlePatternListElementAsImplicit(Context& context) -> void {
  HandlePatternListElement(context, StateKind::Pattern,
                           StateKind::PatternListElementFinishAsImplicit);
}

auto HandleStructPatternDesignatedFieldFinish(Context& context) -> void {
  auto state = context.PopState();
  context.AddNode(NodeKind::StructPatternDesignatedField, state.token,
                  state.has_error);
}

auto HandleStructPatternFieldAfterDesignator(Context& context) -> void {
  auto state = context.PopState();

  auto skip_to_recovery_position = [&](bool add_invalid_parse) {
    auto recovery_pos =
        context.FindNextOf({Lex::TokenKind::Equal, Lex::TokenKind::Comma,
                            Lex::TokenKind::CloseCurlyBrace});

    if (add_invalid_parse) {
      if (context.tokens().GetKind(*recovery_pos) != Lex::TokenKind::Equal) {
        context.AddInvalidParse(*context.position());
      }
    }
    context.SkipTo(*recovery_pos);
  };

  if (state.has_error) {
    // recover from error returned when parsing designator
    skip_to_recovery_position(/*add_invalid_parse=*/false);
  }

  if (!context.PositionIs(Lex::TokenKind::Equal)) {
    if (!state.has_error) {
      state.has_error = true;

      CARBON_DIAGNOSTIC(ExpectedStructPatternDesignatedField, Error,
                        "expected `= value` after `.field`");
      context.emitter().Emit(*context.position(),
                             ExpectedStructPatternDesignatedField);
    }
    skip_to_recovery_position(/*add_invalid_parse=*/true);

    if (context.PositionIs(Lex::TokenKind::Comma) ||
        context.PositionIs(Lex::TokenKind::CloseCurlyBrace)) {
      context.PushState(state, StateKind::StructPatternDesignatedFieldFinish);
      return;
    }
  }

  state.token = context.ConsumeChecked(Lex::TokenKind::Equal);

  context.PushState(state, StateKind::StructPatternDesignatedFieldFinish);
  context.PushStateForPattern(StateKind::Pattern, state.in_var_pattern,
                              state.in_unused_pattern, state.in_struct_pattern,
                              state.binding_context, state.ambient_precedence);
}

auto HandleStructPatternUnderscore(Context& context) -> void {
  auto state = context.PopState();

  if (context.PositionKind(Lookahead::NextToken)
          .is_binding_pattern_operator()) {
    context.PushStateForPattern(StateKind::BindingPattern, state.in_var_pattern,
                                state.in_unused_pattern,
                                state.in_struct_pattern, state.binding_context,
                                state.ambient_precedence);

    return;
  }

  auto underscore = context.ConsumeChecked(Lex::TokenKind::Underscore);

  bool is_last = context.PositionIs(Lex::TokenKind::CloseCurlyBrace);

  if (!is_last) {
    CARBON_DIAGNOSTIC(
        ExpectedCloseAfterUnderscore, Error,
        "unexpected token `{0}` after `_` in struct pattern, expected `}`",
        Lex::TokenKind);
    context.emitter().Emit(*context.position(), ExpectedCloseAfterUnderscore,
                           context.PositionKind());
    state.has_error = true;
  }

  context.AddNode(NodeKind::UnderscoreName, underscore, state.has_error);

  if (state.has_error) {
    context.ReturnErrorOnState();
  }
}

// Handles PatternListElementFinishAs(Tuple|Struct|Explicit|Implicit).
static auto HandlePatternListElementFinish(Context& context,
                                           Lex::TokenKind close_token,
                                           StateKind param_state_kind) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    context.ReturnErrorOnState();
  }

  auto list_token_kind = context.ConsumeListToken(NodeKind::PatternListComma,
                                                  close_token, state.has_error);

  // If we have a comma, the parent is now a tuple pattern not a parenthesized
  // pattern.
  if (list_token_kind != Context::ListTokenKind::Close &&
      param_state_kind == StateKind::PatternListElementAsTuple) {
    auto parent_state = context.PopState();
    CARBON_CHECK(parent_state.kind == StateKind::PatternListFinishAsTuple ||
                 parent_state.kind == StateKind::PatternListFinishAsParen);
    context.PushState(parent_state, StateKind::PatternListFinishAsTuple);
  }

  if (list_token_kind == Context::ListTokenKind::Comma) {
    context.PushStateForPattern(param_state_kind, state.in_var_pattern,
                                state.in_unused_pattern,
                                state.in_struct_pattern, state.binding_context,
                                state.ambient_precedence);
  }
}

auto HandlePatternListElementFinishAsTuple(Context& context) -> void {
  HandlePatternListElementFinish(context, Lex::TokenKind::CloseParen,
                                 StateKind::PatternListElementAsTuple);
}

auto HandlePatternListElementFinishAsStruct(Context& context) -> void {
  HandlePatternListElementFinish(context, Lex::TokenKind::CloseCurlyBrace,
                                 StateKind::PatternListElementAsStruct);
}

auto HandlePatternListElementFinishAsExplicit(Context& context) -> void {
  HandlePatternListElementFinish(context, Lex::TokenKind::CloseParen,
                                 StateKind::PatternListElementAsExplicit);
}

auto HandlePatternListElementFinishAsImplicit(Context& context) -> void {
  HandlePatternListElementFinish(context, Lex::TokenKind::CloseSquareBracket,
                                 StateKind::PatternListElementAsImplicit);
}

// Handles PatternListAs(Tuple|Struct|Explicit|Implicit).
static auto HandlePatternList(Context& context, NodeKind node_kind,
                              Lex::TokenKind open_token_kind,
                              Lex::TokenKind close_token_kind,
                              StateKind param_state,
                              StateKind finish_state_empty,
                              StateKind finish_state_nonempty) -> void {
  auto state = context.PopState();

  if (state.in_struct_pattern) {
    if (node_kind == NodeKind::TuplePatternStart ||
        node_kind == NodeKind::StructPatternStart) {
      CARBON_DIAGNOSTIC(
          NestedPatternListInStructPattern, Error,
          "{0:Tuple|Struct} pattern nested within a struct pattern",
          Diagnostics::BoolAsSelect);
      context.emitter().Emit(*context.position(),
                             NestedPatternListInStructPattern,
                             node_kind == NodeKind::TuplePatternStart);

      context.ReturnErrorOnState();
    }
  }

  if (node_kind == NodeKind::StructPatternStart) {
    state.in_struct_pattern = true;
  }

  auto open_token = context.ConsumeChecked(open_token_kind);
  bool empty = context.PositionIs(close_token_kind);

  context.PushStateForPattern(
      empty ? finish_state_empty : finish_state_nonempty, state.in_var_pattern,
      state.in_unused_pattern, state.in_struct_pattern, state.binding_context,
      state.ambient_precedence);
  context.AddLeafNode(node_kind, open_token);

  if (!empty) {
    context.PushStateForPattern(param_state, state.in_var_pattern,
                                state.in_unused_pattern,
                                state.in_struct_pattern, state.binding_context,
                                PrecedenceGroup::ForTopLevelExpr());
  }
}

auto HandlePatternListAsTuple(Context& context) -> void {
  // If the list is nonempty, use PatternListFinishAsParen as the parent. This
  // will be replaced by PatternListFinishAsTuple if we see a comma.
  HandlePatternList(
      context, NodeKind::TuplePatternStart, Lex::TokenKind::OpenParen,
      Lex::TokenKind::CloseParen, StateKind::PatternListElementAsTuple,
      StateKind::PatternListFinishAsTuple, StateKind::PatternListFinishAsParen);
}

auto HandlePatternListAsStruct(Context& context) -> void {
  HandlePatternList(
      context, NodeKind::StructPatternStart, Lex::TokenKind::OpenCurlyBrace,
      Lex::TokenKind::CloseCurlyBrace, StateKind::PatternListElementAsStruct,
      StateKind::PatternListFinishAsStruct,
      StateKind::PatternListFinishAsStruct);
}

auto HandlePatternListAsExplicit(Context& context) -> void {
  HandlePatternList(context, NodeKind::ExplicitParamListStart,
                    Lex::TokenKind::OpenParen, Lex::TokenKind::CloseParen,
                    StateKind::PatternListElementAsExplicit,
                    StateKind::PatternListFinishAsExplicit,
                    StateKind::PatternListFinishAsExplicit);
}

auto HandlePatternListAsImplicit(Context& context) -> void {
  HandlePatternList(context, NodeKind::ImplicitParamListStart,
                    Lex::TokenKind::OpenSquareBracket,
                    Lex::TokenKind::CloseSquareBracket,
                    StateKind::PatternListElementAsImplicit,
                    StateKind::PatternListFinishAsImplicit,
                    StateKind::PatternListFinishAsImplicit);
}

// Handles PatternListFinishAs(Paren|Tuple|Explicit|Implicit).
static auto HandlePatternListFinish(Context& context, NodeKind node_kind,
                                    Lex::TokenKind token_kind) -> void {
  auto state = context.PopState();

  context.AddNode(node_kind, context.ConsumeChecked(token_kind),
                  state.has_error);
}

auto HandlePatternListFinishAsParen(Context& context) -> void {
  HandlePatternListFinish(context, NodeKind::ParenPattern,
                          Lex::TokenKind::CloseParen);
}

auto HandlePatternListFinishAsTuple(Context& context) -> void {
  HandlePatternListFinish(context, NodeKind::TuplePattern,
                          Lex::TokenKind::CloseParen);
}

auto HandlePatternListFinishAsStruct(Context& context) -> void {
  HandlePatternListFinish(context, NodeKind::StructPattern,
                          Lex::TokenKind::CloseCurlyBrace);
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
