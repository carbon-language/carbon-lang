// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_context.h"

namespace Carbon {

// Handles DeclarationNameAndParamsAs(Optional|Required).
static auto ParserHandleDeclarationNameAndParams(ParserContext& context,
                                                 bool params_required) -> void {
  auto state = context.PopState();

  if (!context.ConsumeAndAddLeafNodeIf(TokenKind::Identifier,
                                       ParseNodeKind::DeclaredName)) {
    context.emitter().Emit(*context.position(), ExpectedDeclarationName,
                           context.tokens().GetKind(state.token));
    context.ReturnErrorOnState();
    return;
  }

  if (context.PositionIs(TokenKind::OpenSquareBracket)) {
    context.PushState(ParserState::DeclarationNameAndParamsAfterDeduced);
    context.PushState(ParserState::ParameterListAsDeduced);
  } else if (context.PositionIs(TokenKind::OpenParen)) {
    context.PushState(ParserState::ParameterListAsRegular);
  } else if (params_required) {
    CARBON_DIAGNOSTIC(ParametersRequiredByIntroducer, Error,
                      "`{0}` requires a `(` for parameters.", TokenKind);
    context.emitter().Emit(*context.position(), ParametersRequiredByIntroducer,
                           context.tokens().GetKind(state.token));
    context.ReturnErrorOnState();
  }
}

auto ParserHandleDeclarationNameAndParamsAsOptional(ParserContext& context)
    -> void {
  ParserHandleDeclarationNameAndParams(context, /*params_required=*/false);
}

auto ParserHandleDeclarationNameAndParamsAsRequired(ParserContext& context)
    -> void {
  ParserHandleDeclarationNameAndParams(context, /*params_required=*/true);
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
