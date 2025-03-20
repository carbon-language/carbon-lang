// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

// Handles VarAs(Decl|Returned).
static auto HandleVar(Context& context, State finish_state,
                      Lex::TokenIndex returned_token = Lex::TokenIndex::None)
    -> void {
  auto state = context.PopState();

  // The finished variable declaration will start at the `var` or `returned`.
  context.PushState(state, finish_state);

  // TODO: is there a cleaner way to give VarAfterPattern access to the `var`
  // token?
  state.token = *(context.position() - 1);
  context.PushState(state, State::VarAfterPattern);

  if (returned_token.has_value()) {
    context.AddLeafNode(NodeKind::ReturnedModifier, returned_token);
  }

  context.PushStateForPattern(State::Pattern, /*in_var_pattern=*/true);
}

auto HandleVarAsDecl(Context& context) -> void {
  HandleVar(context, State::VarFinishAsDecl);
}

auto HandleVarAsReturned(Context& context) -> void {
  auto returned_token = context.Consume();

  if (!context.PositionIs(Lex::TokenKind::Var)) {
    CARBON_DIAGNOSTIC(ExpectedVarAfterReturned, Error,
                      "expected `var` after `returned`");
    context.emitter().Emit(*context.position(), ExpectedVarAfterReturned);
    context.AddLeafNode(NodeKind::EmptyDecl,
                        context.SkipPastLikelyEnd(returned_token),
                        /*has_error=*/true);
    context.PopAndDiscardState();
    return;
  }

  context.AddLeafNode(NodeKind::VariableIntroducer, context.Consume());
  HandleVar(context, State::VarFinishAsDecl, returned_token);
}

auto HandleVarAsFor(Context& context) -> void {
  auto state = context.PopState();

  // The finished variable declaration will start at the `var`.
  context.PushState(state, State::VarFinishAsFor);

  context.PushState(State::Pattern);
}

auto HandleVarAfterPattern(Context& context) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    if (auto after_pattern =
            context.FindNextOf({Lex::TokenKind::Equal, Lex::TokenKind::Semi})) {
      context.SkipTo(*after_pattern);
    }
  }

  context.AddNode(NodeKind::VariablePattern, state.token, state.has_error);

  if (context.PositionIs(Lex::TokenKind::Equal)) {
    context.AddLeafNode(NodeKind::VariableInitializer,
                        context.ConsumeChecked(Lex::TokenKind::Equal));
    context.PushState(State::Expr);
  }
}

auto HandleVarFinishAsDecl(Context& context) -> void {
  auto state = context.PopState();

  auto end_token = state.token;
  if (context.PositionIs(Lex::TokenKind::Semi)) {
    end_token = context.Consume();
  } else {
    // TODO: Disambiguate between statement and member declaration.
    context.DiagnoseExpectedDeclSemi(Lex::TokenKind::Var);
    state.has_error = true;
    end_token = context.SkipPastLikelyEnd(state.token);
  }
  context.AddNode(NodeKind::VariableDecl, end_token, state.has_error);
}

auto HandleVarFinishAsFor(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::VariablePattern, state.token, state.has_error);

  auto end_token = state.token;
  if (context.PositionIs(Lex::TokenKind::In)) {
    end_token = context.Consume();
  } else if (context.PositionIs(Lex::TokenKind::Colon)) {
    CARBON_DIAGNOSTIC(ExpectedInNotColon, Error,
                      "`:` should be replaced by `in`");
    context.emitter().Emit(*context.position(), ExpectedInNotColon);
    state.has_error = true;
    end_token = context.Consume();
  } else {
    CARBON_DIAGNOSTIC(ExpectedIn, Error,
                      "expected `in` after loop `var` declaration");
    context.emitter().Emit(*context.position(), ExpectedIn);
    state.has_error = true;
  }

  context.AddNode(NodeKind::ForIn, end_token, state.has_error);
}

auto HandleVariablePattern(Context& context) -> void {
  auto state = context.PopState();
  if (state.in_var_pattern) {
    CARBON_DIAGNOSTIC(NestedVar, Error, "`var` nested within another `var`");
    context.emitter().Emit(*context.position(), NestedVar);
    state.has_error = true;
  }
  context.PushState(State::FinishVariablePattern);
  context.ConsumeChecked(Lex::TokenKind::Var);

  context.PushStateForPattern(State::Pattern, /*in_var_pattern=*/true);
}

auto HandleFinishVariablePattern(Context& context) -> void {
  auto state = context.PopState();
  context.AddNode(NodeKind::VariablePattern, state.token, state.has_error);

  // Propagate errors to the parent state so that they can take different
  // actions on invalid patterns.
  if (state.has_error) {
    context.ReturnErrorOnState();
  }
}

}  // namespace Carbon::Parse
