// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

static auto ReportUnrecognizedDecl(Context& context) -> void {
  CARBON_DIAGNOSTIC(UnrecognizedDecl, Error,
                    "Unrecognized declaration introducer.");
  context.emitter().Emit(*context.position(), UnrecognizedDecl);
}

// Handles an unrecognized declaration, adding an error node.
static auto HandleUnrecognizedDecl(Context& context) -> void {
  ReportUnrecognizedDecl(context);
  auto cursor = *context.position();
  auto semi = context.SkipPastLikelyEnd(cursor);
  // If we find a semi, create a invalid tree from the cursor to the semi,
  // otherwise just make an invalid node at the cursor.
  if (semi) {
    auto subtree_start = context.tree().size();
    context.AddLeafNode(NodeKind::InvalidParseStart, cursor,
                        /*has_error=*/true);
    context.AddNode(NodeKind::InvalidParseSubtree, *semi, subtree_start,
                    /*has_error=*/true);
  } else {
    context.AddLeafNode(NodeKind::InvalidParse, cursor, /*has_error=*/true);
  }
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
  switch (position_kind) {
#define CARBON_DECL_MODIFIER_KEYWORD_TOKEN(Name, Spelling) \
  case Lex::TokenKind::Name:
#include "toolchain/lex/token_kind.def"
    {
      context.PushState(State::DeclModifier);
      context.AddLeafNode(NodeKind::Placeholder, *context.position());
      return;
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
      context.AddLeafNode(NodeKind::FunctionIntroducer, context.Consume());
      return;
    }
    case Lex::TokenKind::Interface: {
      context.PushState(State::TypeAfterIntroducerAsInterface);
      context.AddLeafNode(NodeKind::InterfaceIntroducer, context.Consume());
      return;
    }
    case Lex::TokenKind::Namespace: {
      context.PushState(State::Namespace);
      context.AddLeafNode(NodeKind::NamespaceStart, context.Consume());
      return;
    }
    case Lex::TokenKind::Semi: {
      context.AddLeafNode(NodeKind::EmptyDecl, context.Consume());
      return;
    }
    case Lex::TokenKind::Var: {
      context.PushState(State::VarAsDecl);
      context.AddLeafNode(NodeKind::VariableIntroducer, context.Consume());
      return;
    }
    case Lex::TokenKind::Let: {
      context.PushState(State::Let);
      context.AddLeafNode(NodeKind::LetIntroducer, context.Consume());
      return;
    }
    default: {
      break;
    }
  }

  HandleUnrecognizedDecl(context);
}

auto HandleDeclModifier(Context& context) -> void {
  auto state = context.PopState();
  CARBON_CHECK(context.GetNodeKind(state.subtree_start) ==
               NodeKind::Placeholder);

  auto introducer = [&](NodeKind node_kind, State next_state) {
    context.ReplaceLeafNode(state.subtree_start, node_kind, context.Consume());
    // Reuse state here to retain its `subtree_start`
    state.state = next_state;
    context.PushState(state);
  };

  switch (context.PositionKind()) {
    // If we see a declaration modifier keyword token, add it as a leaf node
    // and repeat this state.
#define CARBON_DECL_MODIFIER_KEYWORD_TOKEN(Name, Spelling) \
  case Lex::TokenKind::Name:
#include "toolchain/lex/token_kind.def"
    {
      auto modifier_token = context.Consume();
      context.AddLeafNode(NodeKind::DeclModifierKeyword, modifier_token);
      context.PushState(state);
      return;
    }

    // If we see a declaration introducer keyword token, replace the placeholder
    // node and switch to a state to parse the rest of the declaration.
    // We don't allow namespace or empty declarations here since they
    // can't have modifiers and don't use bracketing parse nodes that would
    // allow a variable number of modifier nodes.
    case Lex::TokenKind::Class: {
      introducer(NodeKind::ClassIntroducer, State::TypeAfterIntroducerAsClass);
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

    default: {
      break;
    }
  }
  // For anything else, report an error and output an invalid parse subtree.
  ReportUnrecognizedDecl(context);
  auto cursor = *context.position();
  context.ReplaceLeafNode(state.subtree_start, NodeKind::InvalidParseStart,
                          cursor, /*has_error=*/true);
  auto semi = context.SkipPastLikelyEnd(cursor);
  // If we find a semi, create a invalid tree from the cursor to the semi,
  // otherwise just make an invalid node at the cursor.
  context.AddNode(NodeKind::InvalidParseSubtree, semi ? *semi : cursor,
                  state.subtree_start, /*has_error=*/true);
}

}  // namespace Carbon::Parse
