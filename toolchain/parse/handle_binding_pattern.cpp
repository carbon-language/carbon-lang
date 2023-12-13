// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

auto HandleBindingPattern(Context& context) -> void {
  auto state = context.PopState();

  // Parameters may have keywords prefixing the pattern. They become the parent
  // for the full BindingPattern.
  if (auto token = context.ConsumeIf(Lex::TokenKind::Template)) {
    context.PushState({.state = State::BindingPatternTemplate,
                       .token = *token,
                       .subtree_start = state.subtree_start});
  }

  if (auto token = context.ConsumeIf(Lex::TokenKind::Addr)) {
    context.PushState({.state = State::BindingPatternAddress,
                       .token = *token,
                       .subtree_start = state.subtree_start});
  }

  // Handle an invalid pattern introducer for parameters and variables.
  auto on_error = [&]() {
    CARBON_DIAGNOSTIC(ExpectedBindingPattern, Error,
                      "Expected binding pattern.");
    context.emitter().Emit(*context.position(), ExpectedBindingPattern);
    // Add a placeholder for the type.
    context.AddLeafNode(NodeKind::InvalidParse, *context.position(),
                        /*has_error=*/true);
    state.has_error = true;
    context.PushState(state, State::BindingPatternFinishAsRegular);
  };

  // The first item should be an identifier or `self`.
  bool has_name = false;
  if (auto identifier = context.ConsumeIf(Lex::TokenKind::Identifier)) {
    context.AddLeafNode(NodeKind::IdentifierName, *identifier);
    has_name = true;
  } else if (auto self =
                 context.ConsumeIf(Lex::TokenKind::SelfValueIdentifier)) {
    // Checking will validate the `self` is only declared in the implicit
    // parameter list of a function.
    context.AddLeafNode(NodeKind::SelfValueName, *self);
    has_name = true;
  }
  if (!has_name) {
    // Add a placeholder for the name.
    context.AddLeafNode(NodeKind::IdentifierName, *context.position(),
                        /*has_error=*/true);
    on_error();
    return;
  }

  if (auto kind = context.PositionKind();
      kind == Lex::TokenKind::Colon || kind == Lex::TokenKind::ColonExclaim) {
    state.state = kind == Lex::TokenKind::Colon
                      ? State::BindingPatternFinishAsRegular
                      : State::BindingPatternFinishAsGeneric;
    // Use the `:` or `:!` for the root node.
    state.token = context.Consume();
    context.PushState(state);
    context.PushStateForExpr(PrecedenceGroup::ForType());
  } else {
    on_error();
    return;
  }
}

// Handles BindingPatternFinishAs(Generic|Regular).
static auto HandleBindingPatternFinish(Context& context, NodeKind node_kind)
    -> void {
  auto state = context.PopState();

  context.AddNode(node_kind, state.token, state.subtree_start, state.has_error);

  // Propagate errors to the parent state so that they can take different
  // actions on invalid patterns.
  if (state.has_error) {
    context.ReturnErrorOnState();
  }
}

auto HandleBindingPatternFinishAsGeneric(Context& context) -> void {
  HandleBindingPatternFinish(context, NodeKind::GenericBindingPattern);
}

auto HandleBindingPatternFinishAsRegular(Context& context) -> void {
  HandleBindingPatternFinish(context, NodeKind::BindingPattern);
}

auto HandleBindingPatternAddress(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::Address, state.token, state.subtree_start,
                  state.has_error);

  // If an error was encountered, propagate it while adding a node.
  if (state.has_error) {
    context.ReturnErrorOnState();
  }
}

auto HandleBindingPatternTemplate(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::Template, state.token, state.subtree_start,
                  state.has_error);

  // If an error was encountered, propagate it while adding a node.
  if (state.has_error) {
    context.ReturnErrorOnState();
  }
}

}  // namespace Carbon::Parse
