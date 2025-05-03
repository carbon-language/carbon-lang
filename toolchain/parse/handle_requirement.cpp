// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

auto HandleRequirementBegin(Context& context) -> void {
  auto state = context.PopState();

  // Peek ahead for `.designator = ...`, and give it special handling.
  if (context.PositionKind() == Lex::TokenKind::Period &&
      context.PositionKind(Lookahead::NextToken) ==
          Lex::TokenKind::Identifier &&
      context.PositionKind(static_cast<Lookahead>(2)) ==
          Lex::TokenKind::Equal) {
    auto period = context.Consume();
    context.AddNode(NodeKind::IdentifierNameNotBeforeParams, context.Consume(),
                    /*has_error=*/false);
    context.AddNode(NodeKind::DesignatorExpr, period, /*has_error=*/false);
    state.token = context.Consume();
    context.PushState(state, StateKind::RequirementOperatorFinish);
  } else {
    context.PushState(StateKind::RequirementOperator);
  }
  context.PushStateForExpr(PrecedenceGroup::ForRequirements());
}

auto HandleRequirementOperator(Context& context) -> void {
  auto state = context.PopState();

  switch (context.PositionKind()) {
    // Accept either `impls` or `==`
    case Lex::TokenKind::Impls:
    case Lex::TokenKind::EqualEqual:
      break;

    // Reject `=` since correct usage is consumed in `HandleRequirementBegin`.
    case Lex::TokenKind::Equal: {
      if (!state.has_error) {
        CARBON_DIAGNOSTIC(
            RequirementEqualAfterNonDesignator, Error,
            "requirement can only use `=` after `.member` designator");
        context.emitter().Emit(*context.position(),
                               RequirementEqualAfterNonDesignator);
      }
      context.ReturnErrorOnState();
      return;
    }
    default: {
      if (!state.has_error) {
        CARBON_DIAGNOSTIC(
            ExpectedRequirementOperator, Error,
            "requirement should use `impls`, `=`, or `==` operator");
        context.emitter().Emit(*context.position(),
                               ExpectedRequirementOperator);
      }
      context.ReturnErrorOnState();
      return;
    }
  }
  state.token = context.Consume();
  context.PushState(state, StateKind::RequirementOperatorFinish);
  context.PushStateForExpr(PrecedenceGroup::ForRequirements());
}

auto HandleRequirementOperatorFinish(Context& context) -> void {
  auto state = context.PopState();

  switch (auto token_kind = context.tokens().GetKind(state.token)) {
    case Lex::TokenKind::Impls: {
      context.AddNode(NodeKind::RequirementImpls, state.token, state.has_error);
      break;
    }
    case Lex::TokenKind::Equal: {
      context.AddNode(NodeKind::RequirementEqual, state.token, state.has_error);
      break;
    }
    case Lex::TokenKind::EqualEqual: {
      context.AddNode(NodeKind::RequirementEqualEqual, state.token,
                      state.has_error);
      break;
    }
    default:
      // RequirementOperatorFinish state is only pushed in
      // HandleRequirementOperator on one of the three requirement operator
      // tokens.
      CARBON_FATAL("Unexpected token kind for requirement operator: {0}",
                   token_kind);
      return;
  }
  if (state.has_error) {
    context.ReturnErrorOnState();
  }
  if (auto token = context.ConsumeIf(Lex::TokenKind::And)) {
    context.AddNode(NodeKind::RequirementAnd, *token, /*has_error=*/false);
    context.PushState(StateKind::RequirementBegin);
  }
}

auto HandleWhereFinish(Context& context) -> void {
  auto state = context.PopState();
  if (state.has_error) {
    context.ReturnErrorOnState();
  }
  context.AddNode(NodeKind::WhereExpr, state.token, state.has_error);
}

}  // namespace Carbon::Parse
