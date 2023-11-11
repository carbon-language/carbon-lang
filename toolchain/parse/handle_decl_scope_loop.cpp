// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

// Handles an unrecognized declaration, adding an error node.
static auto HandleUnrecognizedDecl(Context& context) -> void {
  CARBON_DIAGNOSTIC(UnrecognizedDecl, Error,
                    "Unrecognized declaration introducer.");
  context.emitter().Emit(*context.position(), UnrecognizedDecl);
  auto cursor = *context.position();
  auto semi = context.SkipPastLikelyEnd(cursor);
  // Locate the EmptyDecl at the semi when found, but use the
  // original cursor location for an error when not.
  context.AddLeafNode(NodeKind::EmptyDecl, semi ? *semi : cursor,
                      /*has_error=*/true);
}

auto HandleDeclScopeLoop(Context& context) -> void {
  // This maintains the current state unless we're at the end of the scope.

  switch (auto position_kind = context.PositionKind()) {
    case Lex::TokenKind::CloseCurlyBrace:
    case Lex::TokenKind::EndOfFile: {
      // This is the end of the scope, so the loop state ends.
      context.PopAndDiscardState();
      break;
    }
    // `import` and `package` manage their packaging state.
    case Lex::TokenKind::Import: {
      context.PushState(State::Import);
      break;
    }
    case Lex::TokenKind::Package: {
      context.PushState(State::Package);
      break;
    }
    default:
      // Because a non-packaging keyword was encountered, packaging is complete.
      // Misplaced packaging keywords may lead to this being re-triggered.
      if (context.packaging_state() !=
          Context::PackagingState::AfterNonPackagingDecl) {
        if (!context.first_non_packaging_token().is_valid()) {
          context.set_first_non_packaging_token(*context.position());
        }
        context.set_packaging_state(
            Context::PackagingState::AfterNonPackagingDecl);
      }
      switch (position_kind) {
        // Remaining keywords are only valid after imports are complete, and
        // so all result in a `set_packaging_state` call. Note, this may not
        // always be necessary but is probably cheaper than validating.
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
          context.AddLeafNode(NodeKind::EmptyDecl, context.Consume());
          break;
        }
        case Lex::TokenKind::Var: {
          context.PushState(State::VarAsDecl);
          break;
        }
        case Lex::TokenKind::Let: {
          context.PushState(State::Let);
          break;
        }
        default: {
          HandleUnrecognizedDecl(context);
          break;
        }
      }
  }
}

}  // namespace Carbon::Parse
