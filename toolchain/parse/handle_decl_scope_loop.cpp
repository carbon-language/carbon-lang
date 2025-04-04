// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"
#include "toolchain/parse/node_kind.h"

namespace Carbon::Parse {

// Finishes an invalid declaration, skipping past its end.
static auto FinishAndSkipInvalidDecl(Context& context, int32_t subtree_start)
    -> void {
  auto cursor = *context.position();
  // Output an invalid parse subtree including everything up to the next `;`
  // or end of line.
  context.ReplacePlaceholderNode(subtree_start, NodeKind::InvalidParseStart,
                                 cursor, /*has_error=*/true);
  context.AddNode(NodeKind::InvalidParseSubtree,
                  context.SkipPastLikelyEnd(cursor), /*has_error=*/true);
}

// Prints a diagnostic and calls FinishAndSkipInvalidDecl.
static auto HandleUnrecognizedDecl(Context& context, int32_t subtree_start)
    -> void {
  CARBON_DIAGNOSTIC(UnrecognizedDecl, Error,
                    "unrecognized declaration introducer");
  context.emitter().Emit(*context.position(), UnrecognizedDecl);
  FinishAndSkipInvalidDecl(context, subtree_start);
}

// Replaces the introducer placeholder node, and pushes the introducer state for
// processing.
static auto ApplyIntroducer(Context& context, Context::StateStackEntry state,
                            NodeKind introducer_kind, StateKind next_state_kind)
    -> void {
  context.ReplacePlaceholderNode(state.subtree_start, introducer_kind,
                                 context.Consume());
  // Reuse state here to retain its `subtree_start`.
  context.PushState(state, next_state_kind);
}

namespace {
// The kind of declaration introduced by an introducer keyword.
enum class DeclIntroducerKind : int8_t {
  Unrecognized,
  PackagingDecl,
  NonPackagingDecl,
};

// Information about a keyword that might be an introducer keyword.
struct DeclIntroducerInfo {
  DeclIntroducerKind introducer_kind;
  NodeKind node_kind;
  StateKind state_kind;
};
}  // namespace

static constexpr auto DeclIntroducers = [] {
  DeclIntroducerInfo introducers[] = {
#define CARBON_TOKEN(Name)                              \
  {.introducer_kind = DeclIntroducerKind::Unrecognized, \
   .node_kind = NodeKind::InvalidParse,                 \
   .state_kind = StateKind::Invalid},
#include "toolchain/lex/token_kind.def"
  };
  auto set = [&](Lex::TokenKind token_kind, NodeKind node_kind,
                 StateKind state) {
    introducers[token_kind.AsInt()] = {
        .introducer_kind = DeclIntroducerKind::NonPackagingDecl,
        .node_kind = node_kind,
        .state_kind = state};
  };
  auto set_packaging = [&](Lex::TokenKind token_kind, NodeKind node_kind,
                           StateKind state) {
    introducers[token_kind.AsInt()] = {
        .introducer_kind = DeclIntroducerKind::PackagingDecl,
        .node_kind = node_kind,
        .state_kind = state};
  };

  set(Lex::TokenKind::Adapt, NodeKind::AdaptIntroducer,
      StateKind::AdaptAfterIntroducer);
  set(Lex::TokenKind::Alias, NodeKind::AliasIntroducer, StateKind::Alias);
  set(Lex::TokenKind::Base, NodeKind::BaseIntroducer,
      StateKind::BaseAfterIntroducer);
  set(Lex::TokenKind::Choice, NodeKind::ChoiceIntroducer,
      StateKind::ChoiceIntroducer);
  set(Lex::TokenKind::Class, NodeKind::ClassIntroducer,
      StateKind::TypeAfterIntroducerAsClass);
  set(Lex::TokenKind::Constraint, NodeKind::NamedConstraintIntroducer,
      StateKind::TypeAfterIntroducerAsNamedConstraint);
  set(Lex::TokenKind::Export, NodeKind::ExportIntroducer,
      StateKind::ExportName);
  // TODO: Treat `extend` as a declaration introducer.
  set(Lex::TokenKind::Fn, NodeKind::FunctionIntroducer,
      StateKind::FunctionIntroducer);
  set(Lex::TokenKind::Impl, NodeKind::ImplIntroducer,
      StateKind::ImplAfterIntroducer);
  set(Lex::TokenKind::Interface, NodeKind::InterfaceIntroducer,
      StateKind::TypeAfterIntroducerAsInterface);
  set(Lex::TokenKind::Namespace, NodeKind::NamespaceStart,
      StateKind::Namespace);
  set(Lex::TokenKind::Let, NodeKind::LetIntroducer, StateKind::Let);
  set(Lex::TokenKind::Var, NodeKind::VariableIntroducer, StateKind::VarAsDecl);

  set_packaging(Lex::TokenKind::Package, NodeKind::PackageIntroducer,
                StateKind::Package);
  set_packaging(Lex::TokenKind::Library, NodeKind::LibraryIntroducer,
                StateKind::Library);
  set_packaging(Lex::TokenKind::Import, NodeKind::ImportIntroducer,
                StateKind::Import);
  return std::to_array(introducers);
}();

// Attempts to handle the current token as a declaration introducer.
// Returns true if the current position is a declaration. If we see a
// declaration introducer keyword token, replace the placeholder node and switch
// to a state to parse the rest of the declaration.
static auto TryHandleAsDecl(Context& context, Context::StateStackEntry state,
                            bool saw_modifier) -> bool {
  const auto& info = DeclIntroducers[context.PositionKind().AsInt()];

  switch (info.introducer_kind) {
    case DeclIntroducerKind::Unrecognized: {
      // A `;` with no modifiers is an empty declaration.
      if (!saw_modifier) {
        if (auto loc = context.ConsumeIf(Lex::TokenKind::Semi)) {
          context.ReplacePlaceholderNode(state.subtree_start,
                                         NodeKind::EmptyDecl, *loc);
          return true;
        }
      }
      return false;
    }

    case DeclIntroducerKind::PackagingDecl: {
      // Packaging declarations update the packaging state themselves as needed.
      break;
    }

    case DeclIntroducerKind::NonPackagingDecl: {
      // Because a non-packaging keyword was encountered, packaging is complete.
      // Misplaced packaging keywords may lead to this being re-triggered.
      if (context.packaging_state() !=
          Context::PackagingState::AfterNonPackagingDecl) {
        if (!context.first_non_packaging_token().has_value()) {
          context.set_first_non_packaging_token(*context.position());
        }
        context.set_packaging_state(
            Context::PackagingState::AfterNonPackagingDecl);
      }
      break;
    }
  }

  ApplyIntroducer(context, state, info.node_kind, info.state_kind);
  return true;
}

// Returns true if position_kind could be either an introducer or modifier, and
// should be treated as an introducer.
static auto ResolveAmbiguousTokenAsDeclaration(Context& context,
                                               Lex::TokenKind position_kind)
    -> bool {
  switch (position_kind) {
    case Lex::TokenKind::Base:
    case Lex::TokenKind::Export:
    case Lex::TokenKind::Extend:
    case Lex::TokenKind::Impl:
      // This is an ambiguous token, so now we check what the next token is.

      // We use the macro for modifiers, including introducers which are
      // also modifiers (such as `base`). Other introducer tokens need to be
      // added by hand.
      switch (context.PositionKind(Lookahead::NextToken)) {
        case Lex::TokenKind::Adapt:
        case Lex::TokenKind::Alias:
        case Lex::TokenKind::Class:
        case Lex::TokenKind::Constraint:
        case Lex::TokenKind::Extern:
        case Lex::TokenKind::Fn:
        case Lex::TokenKind::Import:
        case Lex::TokenKind::Interface:
        case Lex::TokenKind::Let:
        case Lex::TokenKind::Library:
        case Lex::TokenKind::Namespace:
        case Lex::TokenKind::Var:
#define CARBON_PARSE_NODE_KIND(Name)
#define CARBON_PARSE_NODE_KIND_TOKEN_MODIFIER(Name) case Lex::TokenKind::Name:
#include "toolchain/parse/node_kind.def"

          return false;

        case Lex::TokenKind::Package:
          // `package.foo` is an expression; any other token after `package` is
          // a `package` introducer.
          return context.PositionKind(static_cast<Lookahead>(2)) ==
                 Lex::TokenKind::Period;

        default:
          return true;
      }
      break;
    default:
      return false;
  }
}

// Returns true if the current position is a modifier, handling it if so.
static auto TryHandleAsModifier(Context& context) -> bool {
  auto position_kind = context.PositionKind();
  if (ResolveAmbiguousTokenAsDeclaration(context, position_kind)) {
    return false;
  }

  switch (position_kind) {
#define CARBON_PARSE_NODE_KIND(Name)
#define CARBON_PARSE_NODE_KIND_TOKEN_MODIFIER(Name)                   \
  case Lex::TokenKind::Name:                                          \
    context.AddLeafNode(NodeKind::Name##Modifier, context.Consume()); \
    return true;
#include "toolchain/parse/node_kind.def"

    case Lex::TokenKind::Extern: {
      auto extern_token = context.Consume();
      if (context.PositionIs(Lex::TokenKind::Library)) {
        // `extern library <owning_library>` syntax.
        context.ParseLibrarySpecifier(/*accept_default=*/true);
        // TODO: Consider error recovery when a non-declaration token is next,
        // like a typo of the library name.
        context.AddNode(NodeKind::ExternModifierWithLibrary, extern_token,
                        /*has_error=*/false);
      } else {
        // `extern` syntax without a library.
        context.AddLeafNode(NodeKind::ExternModifier, extern_token);
      }
      return true;
    }

    default:
      return false;
  }
}

auto HandleDecl(Context& context) -> void {
  auto state = context.PopState();

  // Add a placeholder node, to be replaced by the declaration introducer once
  // it is found.
  context.AddLeafNode(NodeKind::Placeholder, *context.position());

  bool saw_modifier = false;
  while (TryHandleAsModifier(context)) {
    saw_modifier = true;
  }
  if (!TryHandleAsDecl(context, state, saw_modifier)) {
    HandleUnrecognizedDecl(context, state.subtree_start);
  }
}

auto HandleDeclScopeLoop(Context& context) -> void {
  // This maintains the current state unless we're at the end of the scope.
  if (context.PositionIs(Lex::TokenKind::CloseCurlyBrace) ||
      context.PositionIs(Lex::TokenKind::FileEnd)) {
    // This is the end of the scope, so the loop state ends.
    context.PopAndDiscardState();
    return;
  }

  context.PushState(StateKind::Decl);
}

}  // namespace Carbon::Parse
