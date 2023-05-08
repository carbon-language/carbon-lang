// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_context.h"

namespace Carbon {

// Handles an unrecognized declaration, adding an error node.
static auto ParserHandleUnrecognizedDeclaration(ParserContext& context)
    -> void {
  CARBON_DIAGNOSTIC(UnrecognizedDeclaration, Error,
                    "Unrecognized declaration introducer.");
  context.emitter().Emit(*context.position(), UnrecognizedDeclaration);
  auto cursor = *context.position();
  auto semi = context.SkipPastLikelyEnd(cursor);
  // Locate the EmptyDeclaration at the semi when found, but use the
  // original cursor location for an error when not.
  context.AddLeafNode(ParseNodeKind::EmptyDeclaration, semi ? *semi : cursor,
                      /*has_error=*/true);
}

auto ParserHandleDeclarationScopeLoop(ParserContext& context) -> void {
  // This maintains the current state unless we're at the end of the scope.

  switch (context.PositionKind()) {
    case TokenKind::CloseCurlyBrace:
    case TokenKind::EndOfFile: {
      // This is the end of the scope, so the loop state ends.
      context.PopAndDiscardState();
      break;
    }
    case TokenKind::Class: {
      context.PushState(ParserState::TypeIntroducerAsClass);
      break;
    }
    case TokenKind::Constraint: {
      context.PushState(ParserState::TypeIntroducerAsNamedConstraint);
      break;
    }
    case TokenKind::Fn: {
      context.PushState(ParserState::FunctionIntroducer);
      break;
    }
    case TokenKind::Interface: {
      context.PushState(ParserState::TypeIntroducerAsInterface);
      break;
    }
    case TokenKind::Semi: {
      context.AddLeafNode(ParseNodeKind::EmptyDeclaration, context.Consume());
      break;
    }
    case TokenKind::Var: {
      context.PushState(ParserState::VarAsSemicolon);
      break;
    }
    default: {
      ParserHandleUnrecognizedDeclaration(context);
      break;
    }
  }
}

}  // namespace Carbon
