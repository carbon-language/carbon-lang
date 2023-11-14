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

  auto position_kind = context.PositionKind();
  switch (position_kind) {
    case Lex::TokenKind::CloseCurlyBrace:
    case Lex::TokenKind::EndOfFile: {
      // This is the end of the scope, so the loop state ends.
      context.PopAndDiscardState();
      return;
    }
    // `import` and `package` manage their packaging state.
    case Lex::TokenKind::Import: {
      context.PushState(State::Import);
      return;
    }
    case Lex::TokenKind::Package: {
      context.PushState(State::Package);
      return;
    }
    default: {
      break;
    }
  }

  // Because a non-packaging keyword was encountered, packaging is complete.
  // Misplaced packaging keywords may lead to this being re-triggered.
  if (context.packaging_state() !=
      Context::PackagingState::AfterNonPackagingDecl) {
    if (!context.first_non_packaging_token().is_valid()) {
      context.set_first_non_packaging_token(*context.position());
    }
    context.set_packaging_state(Context::PackagingState::AfterNonPackagingDecl);
  }

  // Remaining keywords are only valid after imports are complete, and so all
  // result in a `set_packaging_state` call. Note, this may not always be
  // necessary but is probably cheaper than validating.
  switch (position_kind) {
    case Lex::TokenKind::Abstract:
    case Lex::TokenKind::Base: {
      if (context.PositionIs(Lex::TokenKind::Class, Lookahead::NextToken)) {
        context.PushState(State::TypeAfterIntroducerAsClass);
        auto modifier_token = context.Consume();
        auto class_token = context.Consume();
        context.AddLeafNode(NodeKind::ClassIntroducer, class_token);
        context.AddLeafNode(position_kind == Lex::TokenKind::Abstract
                                ? NodeKind::AbstractModifier
                                : NodeKind::BaseModifier,
                            modifier_token);
        return;
      }
      break;
    }
    case Lex::TokenKind::Class: {
      context.PushState(State::TypeAfterIntroducerAsClass);
      context.AddLeafNode(NodeKind::ClassIntroducer, context.Consume());
      return;
    }
    case Lex::TokenKind::Constraint: {
      context.PushState(State::TypeAfterIntroducerAsNamedConstraint);
      context.AddLeafNode(NodeKind::NamedConstraintIntroducer,
                          context.Consume());
      return;
    }
    case Lex::TokenKind::Fn: {
      context.PushState(State::FunctionIntroducer);
      return;
    }
    case Lex::TokenKind::Interface: {
      context.PushState(State::TypeAfterIntroducerAsInterface);
      context.AddLeafNode(NodeKind::InterfaceIntroducer, context.Consume());
      return;
    }
    case Lex::TokenKind::Namespace: {
      context.PushState(State::Namespace);
      return;
    }
    case Lex::TokenKind::Semi: {
      context.AddLeafNode(NodeKind::EmptyDecl, context.Consume());
      return;
    }
    case Lex::TokenKind::Var: {
      context.PushState(State::VarAsDecl);
      return;
    }
    case Lex::TokenKind::Let: {
      context.PushState(State::Let);
      return;
    }
    default: {
      break;
    }
  }

  HandleUnrecognizedDecl(context);
}

}  // namespace Carbon::Parse
