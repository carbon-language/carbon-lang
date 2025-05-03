// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

auto HandleBindingPattern(Context& context) -> void {
  auto state = context.PopState();

  // An `addr` pattern may wrap the binding, and becomes the parent of the
  // `BindingPattern`.
  if (auto token = context.ConsumeIf(Lex::TokenKind::Addr)) {
    context.PushState({.kind = StateKind::BindingPatternAddr,
                       .in_var_pattern = state.in_var_pattern,
                       .token = *token,
                       .subtree_start = state.subtree_start});
  }

  // Handle an invalid pattern introducer for parameters and variables.
  auto on_error = [&](bool expected_name) {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(ExpectedBindingPattern, Error,
                        "expected {0:name|`:` or `:!`} in binding pattern",
                        Diagnostics::BoolAsSelect);
      context.emitter().Emit(*context.position(), ExpectedBindingPattern,
                             expected_name);
      state.has_error = true;
    }
  };

  // A `template` keyword may precede the name.
  auto template_token = context.ConsumeIf(Lex::TokenKind::Template);

  // The first item should be an identifier or `self`.
  bool has_name = false;
  if (auto identifier = context.ConsumeIf(Lex::TokenKind::Identifier)) {
    context.AddLeafNode(NodeKind::IdentifierNameNotBeforeParams, *identifier);
    has_name = true;
  } else if (auto self =
                 context.ConsumeIf(Lex::TokenKind::SelfValueIdentifier)) {
    // Checking will validate the `self` is only declared in the implicit
    // parameter list of a function.
    context.AddLeafNode(NodeKind::SelfValueName, *self);
    has_name = true;
  } else if (auto underscore = context.ConsumeIf(Lex::TokenKind::Underscore)) {
    context.AddLeafNode(NodeKind::UnderscoreName, *underscore);
    has_name = true;
  }
  if (!has_name) {
    // Add a placeholder for the name.
    context.AddLeafNode(NodeKind::IdentifierNameNotBeforeParams,
                        *context.position(), /*has_error=*/true);
    on_error(/*expected_name=*/true);
  }
  if (auto token_kind = context.PositionKind();
      token_kind == Lex::TokenKind::Colon ||
      token_kind == Lex::TokenKind::ColonExclaim) {
    // Add the wrapper node for the `template` keyword if present.
    if (template_token) {
      if (token_kind != Lex::TokenKind::ColonExclaim && !state.has_error) {
        CARBON_DIAGNOSTIC(ExpectedGenericBindingPatternAfterTemplate, Error,
                          "expected `:!` binding after `template`");
        context.emitter().Emit(*template_token,
                               ExpectedGenericBindingPatternAfterTemplate);
        state.has_error = true;
      }
      context.AddNode(NodeKind::TemplateBindingName, *template_token,
                      state.has_error);
    }

    state.kind = token_kind == Lex::TokenKind::Colon
                     ? StateKind::BindingPatternFinishAsRegular
                     : StateKind::BindingPatternFinishAsGeneric;
    // Use the `:` or `:!` for the root node.
    state.token = context.Consume();
    context.PushState(state);
    context.PushStateForExpr(PrecedenceGroup::ForType());
  } else {
    on_error(/*expected_name=*/false);
    // Add a substitute for a type node.
    context.AddInvalidParse(*context.position());
    context.PushState(state, StateKind::BindingPatternFinishAsRegular);
  }
}

// Handles BindingPatternFinishAs(Generic|Regular).
static auto HandleBindingPatternFinish(Context& context, bool is_compile_time)
    -> void {
  auto state = context.PopState();

  auto node_kind = NodeKind::InvalidParse;
  if (state.in_var_pattern) {
    node_kind = NodeKind::VarBindingPattern;
    if (is_compile_time) {
      CARBON_DIAGNOSTIC(CompileTimeBindingInVarDecl, Error,
                        "`var` pattern cannot declare a compile-time binding");
      context.emitter().Emit(*context.position(), CompileTimeBindingInVarDecl);
      state.has_error = true;
    }
  } else {
    if (is_compile_time) {
      node_kind = NodeKind::CompileTimeBindingPattern;
    } else {
      node_kind = NodeKind::LetBindingPattern;
    }
  }
  context.AddNode(node_kind, state.token, state.has_error);

  // Propagate errors to the parent state so that they can take different
  // actions on invalid patterns.
  if (state.has_error) {
    context.ReturnErrorOnState();
  }
}

auto HandleBindingPatternFinishAsGeneric(Context& context) -> void {
  HandleBindingPatternFinish(context, /*is_compile_time=*/true);
}

auto HandleBindingPatternFinishAsRegular(Context& context) -> void {
  HandleBindingPatternFinish(context, /*is_compile_time=*/false);
}

auto HandleBindingPatternAddr(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::Addr, state.token, state.has_error);

  // If an error was encountered, propagate it while adding a node.
  if (state.has_error) {
    context.ReturnErrorOnState();
  }
}

}  // namespace Carbon::Parse
