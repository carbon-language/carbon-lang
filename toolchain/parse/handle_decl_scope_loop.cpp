// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/node_kind.h"

namespace Carbon::Parse {

// Handles positions which are end of scope and packaging declarations. Returns
// true when either applies. When the position is neither, returns false, and
// may still update packaging state.
static auto TryHandleEndOrPackagingDecl(Context& context) -> bool {
  switch (context.PositionKind()) {
    case Lex::TokenKind::CloseCurlyBrace:
    case Lex::TokenKind::FileEnd: {
      // This is the end of the scope, so the loop state ends.
      context.PopAndDiscardState();
      return true;
    }
    // `import`, `library`, and `package` manage their packaging state.
    case Lex::TokenKind::Import: {
      context.PushState(State::Import);
      return true;
    }
    case Lex::TokenKind::Library: {
      context.PushState(State::Library);
      return true;
    }
    case Lex::TokenKind::Package: {
      context.PushState(State::Package);
      return true;
    }
    default: {
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
      return false;
    }
  }
}

// Finishes an invalid declaration, skipping past its end.
static auto FinishAndSkipInvalidDecl(Context& context, int32_t subtree_start)
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

// Prints a diagnostic and calls FinishAndSkipInvalidDecl.
static auto HandleUnrecognizedDecl(Context& context, int32_t subtree_start)
    -> void {
  CARBON_DIAGNOSTIC(UnrecognizedDecl, Error,
                    "Unrecognized declaration introducer.");
  context.emitter().Emit(*context.position(), UnrecognizedDecl);
  FinishAndSkipInvalidDecl(context, subtree_start);
}

// Handles `base` as a declaration.
static auto HandleBaseAsDecl(Context& context, Context::StateStackEntry state)
    -> void {
  // `base` may be followed by:
  // - a colon
  //   => assume it is an introducer, as in `extend base: BaseType;`.
  // - a modifier or an introducer
  //   => assume it is a modifier, as in `base class`; which is handled
  //      by falling through to the next case.
  // Anything else is an error.
  if (context.PositionIs(Lex::TokenKind::Colon, Lookahead::NextToken)) {
    context.ReplacePlaceholderNode(state.subtree_start,
                                   NodeKind::BaseIntroducer, context.Consume());
    // Reuse state here to retain its `subtree_start`
    state.state = State::BaseDecl;
    context.PushState(state);
    context.PushState(State::Expr);
    context.AddLeafNode(NodeKind::BaseColon, context.Consume());
  } else {
    // TODO: If the next token isn't a colon or `class`, try to recover
    // based on whether we're in a class, whether we have an `extend`
    // modifier, and the following tokens.
    context.AddLeafNode(NodeKind::InvalidParse, context.Consume(),
                        /*has_error=*/true);
    CARBON_DIAGNOSTIC(ExpectedAfterBase, Error,
                      "`class` or `:` expected after `base`.");
    context.emitter().Emit(*context.position(), ExpectedAfterBase);
    FinishAndSkipInvalidDecl(context, state.subtree_start);
  }
}

// Returns true if the current position is a declaration. If we see a
// declaration introducer keyword token, replace the placeholder node and switch
// to a state to parse the rest of the declaration.
static auto TryHandleAsDecl(Context& context, Context::StateStackEntry state,
                            bool saw_modifier, Lex::TokenKind position_kind)
    -> bool {
  auto introducer = [&](NodeKind node_kind, State next_state) {
    context.ReplacePlaceholderNode(state.subtree_start, node_kind,
                                   context.Consume());
    // Reuse state here to retain its `subtree_start`.
    state.state = next_state;
    context.PushState(state);
  };

  switch (position_kind) {
    case Lex::TokenKind::Base: {
      HandleBaseAsDecl(context, state);
      return true;
    }
    case Lex::TokenKind::Class: {
      introducer(NodeKind::ClassIntroducer, State::TypeAfterIntroducerAsClass);
      return true;
    }
    case Lex::TokenKind::Constraint: {
      introducer(NodeKind::NamedConstraintIntroducer,
                 State::TypeAfterIntroducerAsNamedConstraint);
      return true;
    }
    case Lex::TokenKind::Extend: {
      // TODO: Treat this `extend` token as a declaration introducer
      HandleUnrecognizedDecl(context, state.subtree_start);
      return true;
    }
    case Lex::TokenKind::Fn: {
      introducer(NodeKind::FunctionIntroducer, State::FunctionIntroducer);
      return true;
    }
    case Lex::TokenKind::Impl: {
      // TODO: Treat this `impl` token as a declaration introducer
      HandleUnrecognizedDecl(context, state.subtree_start);
      return true;
    }
    case Lex::TokenKind::Interface: {
      introducer(NodeKind::InterfaceIntroducer,
                 State::TypeAfterIntroducerAsInterface);
      return true;
    }
    case Lex::TokenKind::Namespace: {
      introducer(NodeKind::NamespaceStart, State::Namespace);
      return true;
    }
    case Lex::TokenKind::Let: {
      introducer(NodeKind::LetIntroducer, State::Let);
      return true;
    }
    case Lex::TokenKind::Var: {
      introducer(NodeKind::VariableIntroducer, State::VarAsDecl);
      return true;
    }

    case Lex::TokenKind::Semi: {
      if (saw_modifier) {
        // Modifiers require an introducer keyword.
        HandleUnrecognizedDecl(context, state.subtree_start);
      } else {
        context.ReplacePlaceholderNode(state.subtree_start, NodeKind::EmptyDecl,
                                       context.Consume());
      }
      return true;
    }

    default:
      return false;
  }
}

// Returns true if position_kind could be either an introducer or modifier, and
// should be treated as an introducer.
static auto ResolveAmbiguousTokenAsDeclaration(Context& context,
                                               Lex::TokenKind position_kind)
    -> bool {
  switch (position_kind) {
    case Lex::TokenKind::Base:
    case Lex::TokenKind::Extend:
    case Lex::TokenKind::Impl:
      // This is an ambiguous token, so now we check what the next token is.

      // We use the macro for modifiers, including introducers which are
      // also modifiers (such as `base`). Other introducer tokens need to be
      // added by hand.
      switch (context.PositionKind(Lookahead::NextToken)) {
        case Lex::TokenKind::Class:
        case Lex::TokenKind::Constraint:
        case Lex::TokenKind::Fn:
        case Lex::TokenKind::Interface:
        case Lex::TokenKind::Let:
        case Lex::TokenKind::Namespace:
        case Lex::TokenKind::Var:
#define CARBON_PARSE_NODE_KIND(...)
#define CARBON_PARSE_NODE_KIND_TOKEN_MODIFIER(Name, ...) \
  case Lex::TokenKind::Name:
#include "toolchain/parse/node_kind.def"
          return false;

        default:
          return true;
      }
      break;
    default:
      return false;
  }
}

// Returns true if the current position is a modifier, handling it if so.
static auto TryHandleAsModifier(Context& context, Lex::TokenKind position_kind)
    -> bool {
  if (ResolveAmbiguousTokenAsDeclaration(context, position_kind)) {
    return false;
  }

  switch (position_kind) {
#define CARBON_PARSE_NODE_KIND(...)
#define CARBON_PARSE_NODE_KIND_TOKEN_MODIFIER(Name, ...)              \
  case Lex::TokenKind::Name:                                          \
    context.AddLeafNode(NodeKind::Name##Modifier, context.Consume()); \
    return true;
#include "toolchain/parse/node_kind.def"

    default:
      return false;
  }
}

auto HandleDeclScopeLoop(Context& context) -> void {
  // This maintains the current state unless we're at the end of the scope.

  if (TryHandleEndOrPackagingDecl(context)) {
    return;
  }

  // Create a state with the correct starting position, with a dummy kind until
  // we see the declaration's introducer.
  context.PushState(State::DeclScopeLoop);
  auto state = context.PopState();

  // Add a placeholder node, to be replaced by the declaration introducer once
  // it is found.
  context.AddLeafNode(NodeKind::Placeholder, *context.position());

  bool saw_modifier = false;
  while (TryHandleAsModifier(context, context.PositionKind())) {
    saw_modifier = true;
  }
  if (!TryHandleAsDecl(context, state, saw_modifier, context.PositionKind())) {
    HandleUnrecognizedDecl(context, state.subtree_start);
  }
}

}  // namespace Carbon::Parse
