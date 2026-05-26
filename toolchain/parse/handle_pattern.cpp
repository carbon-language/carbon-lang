// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

auto HandlePattern(Context& context) -> void {
  auto state = context.PopState();
  switch (context.PositionKind()) {
    case Lex::TokenKind::OpenParen:
      context.PushStateForPattern(StateKind::PatternListAsTuple,
                                  state.in_var_pattern, state.in_unused_pattern,
                                  state.ambient_precedence);
      break;
    case Lex::TokenKind::Var:
      context.PushStateForPattern(StateKind::VariablePattern,
                                  state.in_var_pattern, state.in_unused_pattern,
                                  state.ambient_precedence);
      break;
    case Lex::TokenKind::Unused:
      context.PushStateForPattern(StateKind::UnusedPattern,
                                  state.in_var_pattern, state.in_unused_pattern,
                                  state.ambient_precedence);
      break;
    case Lex::TokenKind::Template:
    case Lex::TokenKind::Ref:
      context.PushStateForPattern(StateKind::BindingPattern,
                                  state.in_var_pattern, state.in_unused_pattern,
                                  state.ambient_precedence);
      break;
    default:
      if (context.PositionKind().is_word() &&
          context.PositionKind(Lookahead::NextToken)
              .is_binding_pattern_operator()) {
        context.PushStateForPattern(
            StateKind::BindingPattern, state.in_var_pattern,
            state.in_unused_pattern, state.ambient_precedence);
        break;
      }
      context.PushState(StateKind::ExprPattern);
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
