// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_context.h"

namespace Carbon {

// Handles DeclarationNameAndParamsAs(Optional|Required).
static auto ParserHandleDeclarationNameAndParams(ParserContext& context,
                                                 ParserState after_name)
    -> void {
  auto state = context.PopState();

  // TODO: Should handle designated names.
  if (auto identifier = context.ConsumeIf(TokenKind::Identifier)) {
    state.state = after_name;
    context.PushState(state);

    if (context.PositionIs(TokenKind::Period)) {
      context.AddLeafNode(ParseNodeKind::Name, *identifier);
      state.state = ParserState::PeriodAsDeclaration;
      context.PushState(state);
    } else {
      context.AddLeafNode(ParseNodeKind::Name, *identifier);
    }
  } else {
    CARBON_DIAGNOSTIC(ExpectedDeclarationName, Error,
                      "`{0}` introducer should be followed by a name.",
                      TokenKind);
    context.emitter().Emit(*context.position(), ExpectedDeclarationName,
                           context.tokens().GetKind(state.token));
    context.ReturnErrorOnState();
    context.AddLeafNode(ParseNodeKind::InvalidParse, *context.position());
  }
}

auto ParserHandleDeclarationNameAndParamsAsNone(ParserContext& context)
    -> void {
  ParserHandleDeclarationNameAndParams(
      context, ParserState::DeclarationNameAndParamsAfterNameAsNone);
}

auto ParserHandleDeclarationNameAndParamsAsOptional(ParserContext& context)
    -> void {
  ParserHandleDeclarationNameAndParams(
      context, ParserState::DeclarationNameAndParamsAfterNameAsOptional);
}

auto ParserHandleDeclarationNameAndParamsAsRequired(ParserContext& context)
    -> void {
  ParserHandleDeclarationNameAndParams(
      context, ParserState::DeclarationNameAndParamsAfterNameAsRequired);
}

enum class Params {
  None,
  Optional,
  Required,
};

static auto ParserHandleDeclarationNameAndParamsAfterName(
    ParserContext& context, Params params) -> void {
  auto state = context.PopState();

  if (context.PositionIs(TokenKind::Period)) {
    // Continue designator processing.
    context.PushState(state);
    state.state = ParserState::PeriodAsDeclaration;
    context.PushState(state);
    return;
  }

  if (params == Params::None) {
    return;
  }

  if (context.PositionIs(TokenKind::OpenSquareBracket)) {
    context.PushState(ParserState::DeclarationNameAndParamsAfterDeduced);
    context.PushState(ParserState::ParameterListAsDeduced);
  } else if (context.PositionIs(TokenKind::OpenParen)) {
    context.PushState(ParserState::ParameterListAsRegular);
  } else if (params == Params::Required) {
    CARBON_DIAGNOSTIC(ParametersRequiredByIntroducer, Error,
                      "`{0}` requires a `(` for parameters.", TokenKind);
    context.emitter().Emit(*context.position(), ParametersRequiredByIntroducer,
                           context.tokens().GetKind(state.token));
    context.ReturnErrorOnState();
  }
}

auto ParserHandleDeclarationNameAndParamsAfterNameAsNone(ParserContext& context)
    -> void {
  ParserHandleDeclarationNameAndParamsAfterName(context, Params::None);
}

auto ParserHandleDeclarationNameAndParamsAfterNameAsOptional(
    ParserContext& context) -> void {
  ParserHandleDeclarationNameAndParamsAfterName(context, Params::Optional);
}

auto ParserHandleDeclarationNameAndParamsAfterNameAsRequired(
    ParserContext& context) -> void {
  ParserHandleDeclarationNameAndParamsAfterName(context, Params::Required);
}

auto ParserHandleDeclarationNameAndParamsAfterDeduced(ParserContext& context)
    -> void {
  context.PopAndDiscardState();

  if (context.PositionIs(TokenKind::OpenParen)) {
    context.PushState(ParserState::ParameterListAsRegular);
  } else {
    CARBON_DIAGNOSTIC(
        ParametersRequiredByDeduced, Error,
        "A `(` for parameters is required after deduced parameters.");
    context.emitter().Emit(*context.position(), ParametersRequiredByDeduced);
    context.ReturnErrorOnState();
  }
}

}  // namespace Carbon
