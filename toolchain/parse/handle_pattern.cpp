// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

// Returns true if `kind` is a token that can follow a variable binding name
// when the `:` and type are erroneously omitted.
static auto CanFollowOmittedTypeInVar(Lex::TokenKind kind) -> bool {
  switch (kind) {
    case Lex::TokenKind::Equal:
    case Lex::TokenKind::EqualGreater:
    case Lex::TokenKind::Semi:
    case Lex::TokenKind::Comma:
    case Lex::TokenKind::CloseParen:
    case Lex::TokenKind::CloseCurlyBrace:
    case Lex::TokenKind::CloseSquareBracket:
    case Lex::TokenKind::Identifier:
    case Lex::TokenKind::SelfTypeIdentifier:
    case Lex::TokenKind::Auto:
    case Lex::TokenKind::Type:
    case Lex::TokenKind::In:
    case Lex::TokenKind::If:
      return true;
    default:
      return kind.is_sized_type_literal();
  }
}

// Returns true if the current token is a binding name in a `var` pattern whose
// `:` and type were erroneously omitted.
static auto IsOmittedTypeVarBinding(Context& context, bool in_var_pattern)
    -> bool {
  if (!in_var_pattern) {
    return false;
  }
  switch (context.PositionKind()) {
    case Lex::TokenKind::Identifier:
    case Lex::TokenKind::Underscore:
      return CanFollowOmittedTypeInVar(
          context.PositionKind(Lookahead::NextToken));
    default:
      return false;
  }
}

// Returns true if the current position starts a binding pattern that is not
// identified by a leading modifier keyword or `self`.
static auto IsUnintroducedBindingPattern(Context& context,
                                         const Context::State& state) -> bool {
  if (state.in_field_shorthand_pattern) {
    return true;
  }
  if (!context.PositionKind().is_word()) {
    return false;
  }
  return context.PositionKind(Lookahead::NextToken)
             .is_binding_pattern_operator() ||
         IsOmittedTypeVarBinding(context, state.in_var_pattern);
}

auto HandlePattern(Context& context) -> void {
  auto state = context.PopState();
  switch (context.PositionKind()) {
    case Lex::TokenKind::OpenParen:
      context.PushStateForPattern(
          StateKind::PatternListAsTuple, state.in_var_pattern,
          state.in_unused_pattern, state.in_field_shorthand_pattern,
          state.binding_context, state.ambient_precedence);
      break;
    case Lex::TokenKind::OpenCurlyBrace:
      context.PushStateForPattern(
          StateKind::PatternListAsStruct, state.in_var_pattern,
          state.in_unused_pattern, state.in_field_shorthand_pattern,
          state.binding_context, state.ambient_precedence);
      break;
    case Lex::TokenKind::Var:
      context.PushStateForPattern(
          StateKind::VariablePattern, state.in_var_pattern,
          state.in_unused_pattern, state.in_field_shorthand_pattern,
          state.binding_context, state.ambient_precedence);
      break;
    case Lex::TokenKind::Unused:
      context.PushStateForPattern(
          StateKind::UnusedPattern, state.in_var_pattern,
          state.in_unused_pattern, state.in_field_shorthand_pattern,
          state.binding_context, state.ambient_precedence);
      break;
    case Lex::TokenKind::Template:
    case Lex::TokenKind::Generic:
    case Lex::TokenKind::Runtime:
    case Lex::TokenKind::Ref:
    // `self` is always a binding, even when its type is omitted (and so is not
    // followed by a `:`).
    case Lex::TokenKind::SelfValueIdentifier:
      context.PushStateForPattern(
          StateKind::BindingPattern, state.in_var_pattern,
          state.in_unused_pattern, state.in_field_shorthand_pattern,
          state.binding_context, state.ambient_precedence);
      break;
    default:
      if (IsUnintroducedBindingPattern(context, state)) {
        context.PushStateForPattern(
            StateKind::BindingPattern, state.in_var_pattern,
            state.in_unused_pattern, state.in_field_shorthand_pattern,
            state.binding_context, state.ambient_precedence);
        break;
      }
      context.PushStateForPattern(
          StateKind::ExprPattern, state.in_var_pattern, state.in_unused_pattern,
          state.in_field_shorthand_pattern, state.binding_context,
          state.ambient_precedence);
      context.PushStateForExpr(state.ambient_precedence);
      break;
  }
}

auto HandleExprPattern(Context& context) -> void {
  auto state = context.PopState();

  // If we parsed an expression followed by a binding operator, we most likely
  // have a malformed attempt to introduce a binding pattern that we interpreted
  // as an expression pattern, so diagnose that here rather than diagnosing a
  // missing `;` at an outer level.
  if (context.PositionKind().is_binding_pattern_operator()) {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(ExpectedBindingName, Error,
                        "unexpected expression before {0} in binding pattern",
                        Lex::TokenKind);
      // TODO: Underline the parsed expression.
      context.emitter().Emit(*context.position(), ExpectedBindingName,
                             context.PositionKind());
      state.has_error = true;
    }
    context.Consume();
    // It'd be nice to skip the type expression here too, but we can't determine
    // the end of it.
  }

  if (state.has_error) {
    context.ReturnErrorOnState();
  }
}

}  // namespace Carbon::Parse
