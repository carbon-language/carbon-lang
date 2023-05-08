// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_context.h"

namespace Carbon {

auto ParserHandleFunctionIntroducer(ParserContext& context) -> void {
  auto state = context.PopState();

  context.AddLeafNode(ParseNodeKind::FunctionIntroducer, context.Consume());

  state.state = ParserState::FunctionAfterParameters;
  context.PushState(state);
  state.state = ParserState::DeclarationNameAndParamsAsRequired;
  context.PushState(state);
}

auto ParserHandleFunctionAfterParameters(ParserContext& context) -> void {
  auto state = context.PopState();

  // Regardless of whether there's a return type, we'll finish the signature.
  state.state = ParserState::FunctionSignatureFinish;
  context.PushState(state);

  // If there is a return type, parse the expression before adding the return
  // type nod.e
  if (context.PositionIs(TokenKind::MinusGreater)) {
    context.PushState(ParserState::FunctionReturnTypeFinish);
    ++context.position();
    context.PushStateForExpression(PrecedenceGroup::ForType());
  }
}

auto ParserHandleFunctionReturnTypeFinish(ParserContext& context) -> void {
  auto state = context.PopState();

  context.AddNode(ParseNodeKind::ReturnType, state.token, state.subtree_start,
                  state.has_error);
}

auto ParserHandleFunctionSignatureFinish(ParserContext& context) -> void {
  auto state = context.PopState();

  switch (context.PositionKind()) {
    case TokenKind::Semi: {
      context.AddNode(ParseNodeKind::FunctionDeclaration, context.Consume(),
                      state.subtree_start, state.has_error);
      break;
    }
    case TokenKind::OpenCurlyBrace: {
      if (auto decl_context = context.GetDeclarationContext();
          decl_context == ParserContext::DeclarationContext::Interface ||
          decl_context == ParserContext::DeclarationContext::NamedConstraint) {
        CARBON_DIAGNOSTIC(
            MethodImplNotAllowed, Error,
            "Method implementations are not allowed in interfaces.");
        context.emitter().Emit(*context.position(), MethodImplNotAllowed);
        context.RecoverFromDeclarationError(state,
                                            ParseNodeKind::FunctionDeclaration,
                                            /*skip_past_likely_end=*/true);
        break;
      }

      context.AddNode(ParseNodeKind::FunctionDefinitionStart, context.Consume(),
                      state.subtree_start, state.has_error);
      // Any error is recorded on the FunctionDefinitionStart.
      state.has_error = false;
      state.state = ParserState::FunctionDefinitionFinish;
      context.PushState(state);
      context.PushState(ParserState::StatementScopeLoop);
      break;
    }
    default: {
      if (!state.has_error) {
        context.emitter().Emit(*context.position(),
                               ExpectedDeclarationSemiOrDefinition,
                               TokenKind::Fn);
      }
      // Only need to skip if we've not already found a new line.
      bool skip_past_likely_end =
          context.tokens().GetLine(*context.position()) ==
          context.tokens().GetLine(state.token);
      context.RecoverFromDeclarationError(
          state, ParseNodeKind::FunctionDeclaration, skip_past_likely_end);
      break;
    }
  }
}

auto ParserHandleFunctionDefinitionFinish(ParserContext& context) -> void {
  auto state = context.PopState();
  context.AddNode(ParseNodeKind::FunctionDefinition, context.Consume(),
                  state.subtree_start, state.has_error);
}

}  // namespace Carbon
