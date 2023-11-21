// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

// Handles an unrecognized declaration, adding an error node.
static auto HandleUnrecognizedDecl(Context& context, int32_t subtree_start)
    -> void {
  CARBON_DIAGNOSTIC(UnrecognizedDecl, Error,
                    "Unrecognized declaration introducer.");
  context.emitter().Emit(*context.position(), UnrecognizedDecl);
  auto cursor = *context.position();
  context.SkipPastLikelyEnd(cursor);
  auto iter = context.position();
  --iter;
  // Output a single invalid parse node if consumed a single token in this
  // error, otherwise output an invalid parse subtree.
  if (*iter == context.tree().node_token(Node(subtree_start))) {
    context.ReplacePlaceholderNode(subtree_start, NodeKind::InvalidParse,
                                   cursor, /*has_error=*/true);
  } else {
    context.ReplacePlaceholderNode(subtree_start, NodeKind::InvalidParseStart,
                                   cursor, /*has_error=*/true);
    // Mark everything up to the last token consumed as invalid.
    context.AddNode(NodeKind::InvalidParseSubtree, *iter, subtree_start,
                    /*has_error=*/true);
  }
}

auto HandleDeclScopeLoop(Context& context) -> void {
  // This maintains the current state unless we're at the end of the scope.

  switch (context.PositionKind()) {
    case Lex::TokenKind::CloseCurlyBrace:
    case Lex::TokenKind::EndOfFile: {
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

  // FIXME: Should we change from:
  //   `ReplacePlaceholderNode` in this function
  // to:
  //   `ReplacePlaceholderNode` in the handlers for the different kinds of
  //   declarations
  // ?

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
      // If we see a declaration modifier keyword token, add it as a leaf node
      // and repeat with the next token.
#define CARBON_DECL_MODIFIER_KEYWORD_TOKEN(Name, Spelling) \
  case Lex::TokenKind::Name:
#include "toolchain/lex/token_kind.def"
      {
        auto modifier_token = context.Consume();
        context.AddLeafNode(NodeKind::DeclModifierKeyword, modifier_token);
        saw_modifier = true;
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
          HandleUnrecognizedDecl(context, state.subtree_start);
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
