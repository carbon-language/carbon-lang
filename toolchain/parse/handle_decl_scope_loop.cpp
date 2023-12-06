// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

static auto OutputInvalidParseSubtree(Context& context, int32_t subtree_start)
    -> void {
  auto cursor = *context.position();
  // Consume to the next `;` or end of line. We ignore the return value since
  // we only care how much was consumed, not whether it ended with a `;`.
  // TODO: adjust the return of SkipPastLikelyEnd or create a new function
  // to avoid going through these hoops.
  context.SkipPastLikelyEnd(cursor);
  // Set `iter` to the last token consumed, one before the current position.
  auto iter = context.position();
  --iter;
  // Output an invalid parse subtree including everything up to the last token
  // consumed.
  context.ReplacePlaceholderNode(subtree_start, NodeKind::InvalidParseStart,
                                 cursor, /*has_error=*/true);
  context.AddNode(NodeKind::InvalidParseSubtree, *iter, subtree_start,
                  /*has_error=*/true);
}

// Handles an unrecognized declaration.
static auto HandleUnrecognizedDecl(Context& context, int32_t subtree_start)
    -> void {
  CARBON_DIAGNOSTIC(UnrecognizedDecl, Error,
                    "Unrecognized declaration introducer.");
  context.emitter().Emit(*context.position(), UnrecognizedDecl);
  OutputInvalidParseSubtree(context, subtree_start);
}

static auto TokenIsModifierOrIntroducer(Lex::TokenKind token_kind) -> bool {
  switch (token_kind) {
    case Lex::TokenKind::Abstract:
    case Lex::TokenKind::Base:
    case Lex::TokenKind::Class:
    case Lex::TokenKind::Constraint:
    case Lex::TokenKind::Default:
    case Lex::TokenKind::Final:
    case Lex::TokenKind::Fn:
    case Lex::TokenKind::Impl:
    case Lex::TokenKind::Interface:
    case Lex::TokenKind::Let:
    case Lex::TokenKind::Private:
    case Lex::TokenKind::Protected:
    case Lex::TokenKind::Var:
    case Lex::TokenKind::Virtual:
      return true;

    default:
      return false;
  }
}

auto HandleDeclScopeLoop(Context& context) -> void {
  // This maintains the current state unless we're at the end of the scope.

  switch (context.PositionKind()) {
    case Lex::TokenKind::CloseCurlyBrace:
    case Lex::TokenKind::FileEnd: {
      // This is the end of the scope, so the loop state ends.
      context.PopAndDiscardState();
      return;
    }
    // `import`, `library`, and `package` manage their packaging state.
    case Lex::TokenKind::Import: {
      context.PushState(State::Import);
      return;
    }
    case Lex::TokenKind::Library: {
      context.PushState(State::Library);
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

  // Create a state with the correct starting position, with a dummy kind until
  // we see the declaration's introducer.
  context.PushState(State::DeclScopeLoop);
  auto state = context.PopState();

  // Add a placeholder node, to be replaced by the declaration introducer once
  // it is found.
  context.AddLeafNode(NodeKind::Placeholder, *context.position());

  auto introducer = [&](NodeKind node_kind, State next_state) {
    context.ReplacePlaceholderNode(state.subtree_start, node_kind,
                                   context.Consume());
    // Reuse state here to retain its `subtree_start`
    state.state = next_state;
    context.PushState(state);
  };

  bool saw_modifier = false;
  while (true) {
    switch (context.PositionKind()) {
      // If we see a access modifier keyword token, add it as a leaf node
      // and repeat with the next token.
      case Lex::TokenKind::Private:
      case Lex::TokenKind::Protected: {
        auto modifier_token = context.Consume();
        context.AddLeafNode(NodeKind::AccessModifierKeyword, modifier_token);
        saw_modifier = true;
        break;
      }

      case Lex::TokenKind::Base:
        // `base` may be followed by:
        // - a colon
        //   => assume it is an introducer, as in `extend base: BaseType;`.
        // - a modifier or an introducer
        //   => assume it is a modifier, as in `base class`; which is handled
        //      by falling through to the next case.
        // Anything else is an error.
        if (context.PositionIs(Lex::TokenKind::Colon, Lookahead::NextToken)) {
          context.ReplacePlaceholderNode(
              state.subtree_start, NodeKind::BaseIntroducer, context.Consume());
          // Reuse state here to retain its `subtree_start`
          state.state = State::BaseDecl;
          context.PushState(state);
          context.PushState(State::Expr);
          context.AddLeafNode(NodeKind::BaseColon, context.Consume());
          return;
        } else if (!TokenIsModifierOrIntroducer(
                       context.PositionKind(Lookahead::NextToken))) {
          // TODO: If the next token isn't a colon or `class`, try to recover
          // based on whether we're in a class, whether we have an `extend`
          // modifier, and the following tokens.
          context.AddLeafNode(NodeKind::InvalidParse, context.Consume(),
                              /*has_error=*/true);
          CARBON_DIAGNOSTIC(ExpectedAfterBase, Error,
                            "`class` or `:` expected after `base`.");
          context.emitter().Emit(*context.position(), ExpectedAfterBase);
          OutputInvalidParseSubtree(context, state.subtree_start);
          return;
        }
        [[fallthrough]];

      // If we see a declaration modifier keyword token, add it as a leaf node
      // and repeat with the next token.
      case Lex::TokenKind::Abstract:
      case Lex::TokenKind::Default:
      case Lex::TokenKind::Final:
      case Lex::TokenKind::Virtual: {
        auto modifier_token = context.Consume();
        context.AddLeafNode(NodeKind::DeclModifierKeyword, modifier_token);
        saw_modifier = true;
        break;
      }

      case Lex::TokenKind::Impl: {
        // `impl` is considered a declaration modifier if it is followed by
        // another modifier or an introducer.
        if (TokenIsModifierOrIntroducer(
                context.PositionKind(Lookahead::NextToken))) {
          context.AddLeafNode(NodeKind::DeclModifierKeyword, context.Consume());
          saw_modifier = true;
        } else {
          // TODO: Treat this `impl` token as a declaration introducer
          HandleUnrecognizedDecl(context, state.subtree_start);
          return;
        }
        break;
      }

      // If we see a declaration introducer keyword token, replace the
      // placeholder node and switch to a state to parse the rest of the
      // declaration. We don't allow namespace or empty declarations here since
      // they can't have modifiers and don't use bracketing parse nodes that
      // would allow a variable number of modifier nodes.
      case Lex::TokenKind::Class: {
        introducer(NodeKind::ClassIntroducer,
                   State::TypeAfterIntroducerAsClass);
        return;
      }
      case Lex::TokenKind::Constraint: {
        introducer(NodeKind::NamedConstraintIntroducer,
                   State::TypeAfterIntroducerAsNamedConstraint);
        return;
      }
      case Lex::TokenKind::Fn: {
        introducer(NodeKind::FunctionIntroducer, State::FunctionIntroducer);
        return;
      }
      case Lex::TokenKind::Interface: {
        introducer(NodeKind::InterfaceIntroducer,
                   State::TypeAfterIntroducerAsInterface);
        return;
      }
      case Lex::TokenKind::Var: {
        introducer(NodeKind::VariableIntroducer, State::VarAsDecl);
        return;
      }
      case Lex::TokenKind::Let: {
        introducer(NodeKind::LetIntroducer, State::Let);
        return;
      }

      // We don't allow namespace or empty declarations after a modifier since
      // they can't have modifiers and don't use bracketing parse nodes that
      // would allow a variable number of modifier nodes.
      case Lex::TokenKind::Namespace: {
        if (saw_modifier) {
          CARBON_DIAGNOSTIC(NamespaceAfterModifiers, Error,
                            "`namespace` unexpected after modifiers.");
          context.emitter().Emit(*context.position(), NamespaceAfterModifiers);
          OutputInvalidParseSubtree(context, state.subtree_start);
        } else {
          introducer(NodeKind::NamespaceStart, State::Namespace);
        }
        return;
      }
      case Lex::TokenKind::Semi: {
        if (saw_modifier) {
          HandleUnrecognizedDecl(context, state.subtree_start);
        } else {
          context.ReplacePlaceholderNode(
              state.subtree_start, NodeKind::EmptyDecl, context.Consume());
        }
        return;
      }

      default: {
        // For anything else, report an error and output an invalid parse node
        // or subtree.
        HandleUnrecognizedDecl(context, state.subtree_start);
        return;
      }
    }
  }
}

}  // namespace Carbon::Parse
