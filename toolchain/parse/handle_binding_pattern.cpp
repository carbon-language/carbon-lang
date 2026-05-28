// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

auto HandleBindingPattern(Context& context) -> void {
  auto state = context.PopState();

  // Handle an invalid pattern introducer for parameters and variables.
  auto on_error = [&](bool expected_name, bool recover_as_raw = false) {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(
          ExpectedBindingPattern, Error,
          "expected {0:name|`:`, `:!`, or `:?`} in binding pattern"
          "{1:; prefix reserved word with `r#` to form a valid identifier|}",
          Diagnostics::BoolAsSelect, Diagnostics::BoolAsSelect);
      context.emitter().Emit(*context.position(), ExpectedBindingPattern,
                             expected_name, recover_as_raw);
      state.has_error = !recover_as_raw;
    }
  };

  // A `template` keyword may precede the name.
  auto template_token = context.ConsumeIf(Lex::TokenKind::Template);
  auto ref_token = context.ConsumeIf(Lex::TokenKind::Ref);
  if (ref_token && state.in_var_pattern) {
    CARBON_DIAGNOSTIC(RefInsideVar, Error, "found `ref` inside `var` pattern");
    context.emitter().Emit(*ref_token, RefInsideVar);
    state.has_error = true;
  }

  // The first item should be an identifier, the placeholder `_`, or `self`.
  if (auto identifier = context.ConsumeIf(Lex::TokenKind::Identifier)) {
    context.AddLeafNode(NodeKind::IdentifierNameNotBeforeSignature,
                        *identifier);
  } else if (auto self =
                 context.ConsumeIf(Lex::TokenKind::SelfValueIdentifier)) {
    // Checking will validate the `self` is only declared in the implicit
    // parameter list of a function.
    context.AddLeafNode(NodeKind::SelfValueName, *self);
  } else if (auto underscore = context.ConsumeIf(Lex::TokenKind::Underscore)) {
    context.AddLeafNode(NodeKind::UnderscoreName, *underscore);
  } else if (context.PositionKind().is_word() &&
             context.PositionKind(Lookahead::NextToken)
                 .is_binding_pattern_operator()) {
    // A word token that is not a valid binding name appeared before the `:`,
    // such as a numeric type literal or a keyword. For error recovery, convert
    // the token to an identifier, as we can be confident that a word in this
    // position was intended to be a declared name.
    auto word_as_identifier =
        context.tokens().AddPostLexingRecoveryTokenAsIdentifier(
            context.Consume());
    context.AddLeafNode(NodeKind::IdentifierNameNotBeforeSignature,
                        word_as_identifier);
    on_error(/*expected_name=*/true, /*recover_as_raw*/ true);
  } else {
    // Add a placeholder for the name.
    context.AddLeafNode(NodeKind::IdentifierNameNotBeforeSignature,
                        *context.position(), /*has_error=*/true);
    on_error(/*expected_name=*/true);
  }

  if (auto token_kind = context.PositionKind();
      token_kind.is_binding_pattern_operator()) {
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
    if (ref_token) {
      if (token_kind != Lex::TokenKind::Colon && !state.has_error) {
        CARBON_DIAGNOSTIC(ExpectedRuntimeBindingPatternAfterRef, Error,
                          "expected `:` binding after `ref`");
        context.emitter().Emit(*ref_token,
                               ExpectedRuntimeBindingPatternAfterRef);
        state.has_error = true;
      }
      context.AddNode(NodeKind::RefBindingName, *ref_token, state.has_error);
    }

    switch (token_kind) {
      case Lex::TokenKind::Colon:
        state.kind = StateKind::BindingPatternFinishAsRegular;
        break;
      case Lex::TokenKind::ColonExclaim:
        state.kind = StateKind::BindingPatternFinishAsGeneric;
        break;
      case Lex::TokenKind::ColonQuestion:
        state.kind = StateKind::BindingPatternFinishAsForm;
        break;
      default:
        CARBON_FATAL("Unexpected token kind");
    }

    // Use the `:`, `:!`, or `:?` for the root node.
    state.token = context.Consume();

    if (token_kind == Lex::TokenKind::ColonExclaim) {
      // Add a virtual node before the compile time binding's type
      // expression.
      context.AddNode(NodeKind::CompileTimeBindingPatternStart, state.token,
                      state.has_error);
    }

    context.PushState(state);
    context.PushStateForExpr(PrecedenceGroup::ForType());
  } else {
    on_error(/*expected_name=*/false);
    // Add a substitute for a type node.
    context.AddInvalidParse(*context.position());
    context.PushState(state, StateKind::BindingPatternFinishAsRegular);
  }
}

// Handles BindingPatternFinishAs(Generic|Regular|Form).
static auto HandleBindingPatternFinish(Context& context, StateKind finish_kind)
    -> void {
  auto state = context.PopState();

  auto node_kind = NodeKind::InvalidParse;
  if (state.in_var_pattern) {
    node_kind = NodeKind::VarBindingPattern;
    if (finish_kind != StateKind::BindingPatternFinishAsRegular) {
      CARBON_DIAGNOSTIC(NonRegularBindingInVarDecl, Error,
                        "found {0:`:!`|`:?`} pattern inside `var` pattern",
                        Diagnostics::BoolAsSelect);
      context.emitter().Emit(
          *context.position(), NonRegularBindingInVarDecl,
          finish_kind == StateKind::BindingPatternFinishAsGeneric);
      state.has_error = true;
    }
  } else {
    switch (finish_kind) {
      case StateKind::BindingPatternFinishAsGeneric:
        node_kind = NodeKind::CompileTimeBindingPattern;
        break;
      case StateKind::BindingPatternFinishAsRegular:
        node_kind = NodeKind::LetBindingPattern;
        break;
      case StateKind::BindingPatternFinishAsForm:
        node_kind = NodeKind::FormBindingPattern;
        break;
      default:
        CARBON_FATAL("Unexpected StateKind {0}", finish_kind);
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
  HandleBindingPatternFinish(context, StateKind::BindingPatternFinishAsGeneric);
}

auto HandleBindingPatternFinishAsRegular(Context& context) -> void {
  HandleBindingPatternFinish(context, StateKind::BindingPatternFinishAsRegular);
}

auto HandleBindingPatternFinishAsForm(Context& context) -> void {
  HandleBindingPatternFinish(context, StateKind::BindingPatternFinishAsForm);
}

}  // namespace Carbon::Parse
