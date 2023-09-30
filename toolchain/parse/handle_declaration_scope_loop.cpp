// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

// Handles an unrecognized declaration, adding an error node.
static auto HandleUnrecognizedDeclaration(Context& context) -> void {
  CARBON_DIAGNOSTIC(UnrecognizedDeclaration, Error,
                    "Unrecognized declaration introducer.");
  context.emitter().Emit(*context.position(), UnrecognizedDeclaration);
  auto cursor = *context.position();
  auto semi = context.SkipPastLikelyEnd(cursor);
  // Locate the EmptyDeclaration at the semi when found, but use the
  // original cursor location for an error when not.
  context.AddLeafNode(NodeKind::EmptyDeclaration, semi ? *semi : cursor,
                      /*has_error=*/true);
}

auto HandleDeclarationScopeLoop(Context& context) -> void {
  // This maintains the current state unless we're at the end of the scope.

  switch (context.PositionKind()) {
    case Lex::TokenKind::CloseCurlyBrace:
    case Lex::TokenKind::EndOfFile: {
      // This is the end of the scope, so the loop state ends.
      context.PopAndDiscardState();
      break;
    }
    case Lex::TokenKind::Class: {
      context.PushState(State::TypeIntroducerAsClass);
      break;
    }
    case Lex::TokenKind::Constraint: {
      context.PushState(State::TypeIntroducerAsNamedConstraint);
      break;
    }
    case Lex::TokenKind::Fn: {
      context.PushState(State::FunctionIntroducer);
      break;
    }
    case Lex::TokenKind::Interface: {
      context.PushState(State::TypeIntroducerAsInterface);
      break;
    }
    case Lex::TokenKind::Namespace: {
      context.PushState(State::Namespace);
      break;
    }
    case Lex::TokenKind::Semi: {
      context.AddLeafNode(NodeKind::EmptyDeclaration, context.Consume());
      break;
    }
    case Lex::TokenKind::Var: {
      context.PushState(State::VarAsSemicolon);
      break;
    }
    case Lex::TokenKind::Let: {
      context.PushState(State::Let);
      break;
    }
    default: {
      HandleUnrecognizedDeclaration(context);
      break;
    }
  }
}

}  // namespace Carbon::Parse
