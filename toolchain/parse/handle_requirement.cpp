// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/handle_requirement.h"

#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

auto BeginRequirement(Context& context) -> void {
  context.PushState(State::RequirementOperator);
  context.PushStateForExpr(PrecedenceGroup::ForRequirements());
}

auto HandleRequirementOperator(Context& context) -> void {
  auto state = context.PopState();
  auto kind = context.PositionKind();
  state.token = context.Consume();

  switch (kind) {
    case Lex::TokenKind::Impls: {
      break;
    }
    case Lex::TokenKind::Equal: {
      break;
    }
    case Lex::TokenKind::EqualEqual: {
      break;
    }
    default: {
      CARBON_DIAGNOSTIC(
          ExpectedRequirementOperator, Error,
          "requirement should use `impls`, `=`, or `==` operator.");
      context.emitter().Emit(state.token, ExpectedRequirementOperator);
      context.ReturnErrorOnState();
      return;
    }
  }
  context.PushState(state, State::RequirementOperatorFinish);
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
      context.AddNode(NodeKind::RequirementAssign, state.token,
                      state.has_error);
      break;
    }
    case Lex::TokenKind::EqualEqual: {
      context.AddNode(NodeKind::RequirementEquals, state.token,
                      state.has_error);
      break;
    }
    default:
      // RequirementOperatorFinish state is only pushed in
      // HandleRequirementOperator on one of the three requirement operator
      // tokens.
      CARBON_FATAL() << "Unexpected token kind for requirement operator: "
                     << token_kind;
      return;
  }
  if (state.has_error) {
    context.ReturnErrorOnState();
  }
  if (auto token = context.ConsumeIf(Lex::TokenKind::And)) {
    context.AddNode(NodeKind::RequirementAnd, *token, /*has_error=*/false);
    context.PushState(State::RequirementOperator);
    context.PushStateForExpr(PrecedenceGroup::ForRequirements());
  }
}

auto HandleWhereFinish(Context& context) -> void {
  auto state = context.PopState();
  context.AddNode(NodeKind::WhereExpr, state.token, state.has_error);
}

}  // namespace Carbon::Parse
