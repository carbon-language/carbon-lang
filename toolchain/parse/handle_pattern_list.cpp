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

  context.PushStateForPattern(finish_state_kind, state.in_var_pattern,
                              state.in_unused_pattern,
                              state.ambient_precedence);
  context.PushStateForPattern(pattern_state, state.in_var_pattern,
                              state.in_unused_pattern,
                              state.ambient_precedence);
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
    case Lex::TokenKind::Var:
      HandlePatternListElement(context, StateKind::VariablePattern,
                               StateKind::PatternListElementFinishAsStruct);
      break;
    case Lex::TokenKind::Let:
    case Lex::TokenKind::Ref:
    case Lex::TokenKind::Identifier:
      HandlePatternListElement(context, StateKind::BindingPattern,
                               StateKind::PatternListElementFinishAsStruct);
      break;
    default:
      auto state = context.PopState();
      state.has_error = true;

      CARBON_DIAGNOSTIC(ExpectedStructPatternField, Error,
                        "expected `.field = value` or Binding Pattern");

      context.emitter().Emit(*context.position(), ExpectedStructPatternField);

      auto recovery_pos =
          context.FindNextOf({Lex::TokenKind::Equal, Lex::TokenKind::Comma});

      if (!recovery_pos ||
          context.tokens().GetKind(*recovery_pos) == Lex::TokenKind::Comma) {
        context.PushState(state, StateKind::PatternListElementFinishAsStruct);
        break;
      }

      context.SkipTo(*recovery_pos);

      if (context.PositionIs(Lex::TokenKind::Equal)) {
        state.token = context.ConsumeChecked(Lex::TokenKind::Equal);

        context.PushState(state, StateKind::StructPatternDesignatedFieldFinish);
        context.PushStateForPattern(StateKind::Pattern, state.in_var_pattern,
                                    state.in_unused_pattern,
                                    state.ambient_precedence);
      } else {
        context.PushState(state, StateKind::PatternListElementFinishAsStruct);
      }

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

  if (state.has_error) {
    context.AddLeafNode(NodeKind::InvalidParse, state.token,
                        /*has_error=*/true);
    context.ReturnErrorOnState();
  } else {
    context.AddNode(NodeKind::StructPatternDesignatedField, state.token,
                    state.has_error);
  }
}

auto HandleStructPatternFieldAfterDesignator(Context& context) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    auto recovery_pos =
        context.FindNextOf({Lex::TokenKind::Equal, Lex::TokenKind::Comma});
    if (!recovery_pos ||
        context.tokens().GetKind(*recovery_pos) == Lex::TokenKind::Comma) {
      context.PushState(state, StateKind::StructPatternDesignatedFieldFinish);
      return;
    }
    context.SkipTo(*recovery_pos);
  }

  if (!context.PositionIs(Lex::TokenKind::Equal)) {
    CARBON_DIAGNOSTIC(ExpectedStructPatternDesignatedField, Error,
                      "expected `.field = value`");

    context.emitter().Emit(*context.position(),
                           ExpectedStructPatternDesignatedField);

    state.has_error = true;
    context.PushState(state, StateKind::StructPatternDesignatedFieldFinish);

    return;
  }

  state.token = context.ConsumeChecked(Lex::TokenKind::Equal);

  context.PushState(state, StateKind::StructPatternDesignatedFieldFinish);
  context.PushStateForPattern(StateKind::Pattern, state.in_var_pattern,
                              state.in_unused_pattern,
                              state.ambient_precedence);
}

auto HandleStructPatternUnderscore(Context& context) -> void {
  auto state = context.PopState();
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
  auto open_token = context.ConsumeChecked(open_token_kind);
  bool empty = context.PositionIs(close_token_kind);

  context.PushStateForPattern(
      empty ? finish_state_empty : finish_state_nonempty, state.in_var_pattern,
      state.in_unused_pattern, state.ambient_precedence);
  context.AddLeafNode(node_kind, open_token);

  if (!empty) {
    context.PushStateForPattern(param_state, state.in_var_pattern,
                                state.in_unused_pattern,
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
