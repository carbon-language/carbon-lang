// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

// Handles DeclarationNameAndParamsAs(Optional|Required).
static auto HandleDeclarationNameAndParams(Context& context, State after_name)
    -> void {
  auto state = context.PopState();

  // TODO: Should handle designated names.
  if (auto identifier = context.ConsumeIf(Lex::TokenKind::Identifier)) {
    state.state = after_name;
    context.PushState(state);

    if (context.PositionIs(Lex::TokenKind::Period)) {
      // Because there's a qualifier, we process the first segment as an
      // expression for simplicity. This just means semantics has one less thing
      // to handle here.
      context.AddLeafNode(NodeKind::NameExpression, *identifier);
      state.state = State::PeriodAsDeclaration;
      context.PushState(state);
    } else {
      context.AddLeafNode(NodeKind::Name, *identifier);
    }
  } else {
    CARBON_DIAGNOSTIC(ExpectedDeclarationName, Error,
                      "`{0}` introducer should be followed by a name.",
                      Lex::TokenKind);
    context.emitter().Emit(*context.position(), ExpectedDeclarationName,
                           context.tokens().GetKind(state.token));
    context.ReturnErrorOnState();
    context.AddLeafNode(NodeKind::InvalidParse, *context.position(),
                        /*has_error=*/true);
  }
}

auto HandleDeclarationNameAndParamsAsNone(Context& context) -> void {
  HandleDeclarationNameAndParams(
      context, State::DeclarationNameAndParamsAfterNameAsNone);
}

auto HandleDeclarationNameAndParamsAsOptional(Context& context) -> void {
  HandleDeclarationNameAndParams(
      context, State::DeclarationNameAndParamsAfterNameAsOptional);
}

auto HandleDeclarationNameAndParamsAsRequired(Context& context) -> void {
  HandleDeclarationNameAndParams(
      context, State::DeclarationNameAndParamsAfterNameAsRequired);
}

enum class Params : int8_t {
  None,
  Optional,
  Required,
};

static auto HandleDeclarationNameAndParamsAfterName(Context& context,
                                                    Params params) -> void {
  auto state = context.PopState();

  if (context.PositionIs(Lex::TokenKind::Period)) {
    // Continue designator processing.
    context.PushState(state);
    state.state = State::PeriodAsDeclaration;
    context.PushState(state);
    return;
  }

  if (params == Params::None) {
    return;
  }

  if (context.PositionIs(Lex::TokenKind::OpenSquareBracket)) {
    context.PushState(State::DeclarationNameAndParamsAfterDeduced);
    context.PushState(State::ParameterListAsDeduced);
  } else if (context.PositionIs(Lex::TokenKind::OpenParen)) {
    context.PushState(State::ParameterListAsRegular);
  } else if (params == Params::Required) {
    CARBON_DIAGNOSTIC(ParametersRequiredByIntroducer, Error,
                      "`{0}` requires a `(` for parameters.", Lex::TokenKind);
    context.emitter().Emit(*context.position(), ParametersRequiredByIntroducer,
                           context.tokens().GetKind(state.token));
    context.ReturnErrorOnState();
  }
}

auto HandleDeclarationNameAndParamsAfterNameAsNone(Context& context) -> void {
  HandleDeclarationNameAndParamsAfterName(context, Params::None);
}

auto HandleDeclarationNameAndParamsAfterNameAsOptional(Context& context)
    -> void {
  HandleDeclarationNameAndParamsAfterName(context, Params::Optional);
}

auto HandleDeclarationNameAndParamsAfterNameAsRequired(Context& context)
    -> void {
  HandleDeclarationNameAndParamsAfterName(context, Params::Required);
}

auto HandleDeclarationNameAndParamsAfterDeduced(Context& context) -> void {
  context.PopAndDiscardState();

  if (context.PositionIs(Lex::TokenKind::OpenParen)) {
    context.PushState(State::ParameterListAsRegular);
  } else {
    CARBON_DIAGNOSTIC(
        ParametersRequiredByDeduced, Error,
        "A `(` for parameters is required after deduced parameters.");
    context.emitter().Emit(*context.position(), ParametersRequiredByDeduced);
    context.ReturnErrorOnState();
  }
}

}  // namespace Carbon::Parse
