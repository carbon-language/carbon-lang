// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

// Handles VarAs(Regular|Returned).
static auto HandleVar(Context& context, StateKind finish_state_kind,
                      Lex::TokenIndex returned_token = Lex::TokenIndex::None)
    -> void {
  auto state = context.PopState();

  // The finished variable declaration will start at the `var` or `returned`.
  context.PushState(state, finish_state_kind);

  // TODO: is there a cleaner way to give VarAfterPattern access to the `var`
  // token?
  state.token = *(context.position() - 1);
  context.PushState(state, StateKind::VarAfterPatternAsVar);

  if (returned_token.has_value()) {
    context.AddLeafNode(NodeKind::ReturnedModifier, returned_token);
  }

  context.PushStateForPattern(StateKind::Pattern, /*in_var_pattern=*/true);
}

auto HandleVarAsRegular(Context& context) -> void {
  HandleVar(context, StateKind::VarFinishAsRegular);
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
  HandleVar(context, StateKind::VarFinishAsRegular, returned_token);
}

auto HandleFieldDecl(Context& context) -> void {
  auto state = context.PopState();

  auto identifier = context.ConsumeIf(Lex::TokenKind::Identifier);
  if (!identifier) {
    CARBON_DIAGNOSTIC(ExpectedFieldIdentifier, Error,
                      "expected identifier in field declaration");
    context.emitter().Emit(*context.position(), ExpectedFieldIdentifier);
  }
  auto colon = context.ConsumeIf(Lex::TokenKind::Colon);
  if (identifier && !colon) {
    CARBON_DIAGNOSTIC(ExpectedFieldColon, Error,
                      "expected `:` in field declaration");
    context.emitter().Emit(*context.position(), ExpectedFieldColon);
  }
  if (!identifier || !colon) {
    context.AddNode(NodeKind::FieldDecl,
                    context.SkipPastLikelyEnd(*(context.position() - 1)),
                    /*has_error=*/true);
    state.has_error = true;
    return;
  }
  context.PushState(state, StateKind::VarFinishAsField);
  context.AddLeafNode(NodeKind::IdentifierNameNotBeforeParams, *identifier);
  state.token = *colon;
  context.PushState(state, StateKind::VarAfterPatternAsField);
  context.PushState(StateKind::Expr);
}

static auto HandleVarAfterPattern(Context& context, NodeKind pattern_kind,
                                  NodeKind init_kind) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    if (auto after_pattern =
            context.FindNextOf({Lex::TokenKind::Equal, Lex::TokenKind::Semi})) {
      context.SkipTo(*after_pattern);
    }
  }

  context.AddNode(pattern_kind, state.token, state.has_error);

  if (context.PositionIs(Lex::TokenKind::Equal)) {
    context.AddLeafNode(init_kind,
                        context.ConsumeChecked(Lex::TokenKind::Equal));
    context.PushState(StateKind::Expr);
  }
}

auto HandleVarAfterPatternAsVar(Context& context) -> void {
  HandleVarAfterPattern(context, NodeKind::VariablePattern,
                        NodeKind::VariableInitializer);
}

auto HandleVarAfterPatternAsField(Context& context) -> void {
  HandleVarAfterPattern(context, NodeKind::FieldNameAndType,
                        NodeKind::FieldInitializer);
}

static auto HandleVarFinish(Context& context, NodeKind node_kind) -> void {
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
  context.AddNode(node_kind, end_token, state.has_error);
}

auto HandleVarFinishAsRegular(Context& context) -> void {
  HandleVarFinish(context, NodeKind::VariableDecl);
}

auto HandleVarFinishAsField(Context& context) -> void {
  HandleVarFinish(context, NodeKind::FieldDecl);
}

auto HandleVariablePattern(Context& context) -> void {
  auto state = context.PopState();
  if (state.in_var_pattern) {
    CARBON_DIAGNOSTIC(NestedVar, Error, "`var` nested within another `var`");
    context.emitter().Emit(*context.position(), NestedVar);
    state.has_error = true;
  }
  context.PushState(StateKind::FinishVariablePattern);
  context.ConsumeChecked(Lex::TokenKind::Var);

  context.PushStateForPattern(StateKind::Pattern, /*in_var_pattern=*/true);
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
