// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

// Determines whether a `:` binding is a generic binding, from its contextual
// default and any explicit phase keyword. `is_form` is true for a `:?` form
// binding, whose phase is fixed and unaffected by phase keywords.
//
// A keyword that is *redundant* with the contextual default is diagnosed here.
// A keyword that is *invalid* for the context (such as `runtime` on a
// compile-time entity's parameter) is intentionally not rejected here: the
// requested phase is honored, and `check` diagnoses the resulting phase as
// invalid for the context and recovers by building an error binding that still
// introduces the name. Either way no parse node is flagged as an error, so
// `check` never aborts on an invalid parse tree; the caller preserves an
// explicit `runtime` keyword as a `RuntimeBindingName` node so its token is
// accounted for and `check` can see the requested phase.
static auto ResolveBindingPhase(Context& context, Context::State& state,
                                bool is_form,
                                std::optional<Lex::TokenIndex> template_token,
                                std::optional<Lex::TokenIndex> generic_token,
                                std::optional<Lex::TokenIndex> runtime_token,
                                bool& redundant_modifier) -> bool {
  // `template`/`generic` force a generic binding, `runtime` forces a runtime
  // binding, and otherwise the context's default applies.
  bool resolved_generic;
  if (template_token || generic_token) {
    resolved_generic = true;
  } else if (runtime_token) {
    resolved_generic = false;
  } else {
    resolved_generic = state.binding_context != BindingContext::ExplicitParam;
  }

  // A form binding's phase is fixed, so phase keywords don't apply to it, and
  // there is no point diagnosing a redundant keyword once the binding is
  // already in error.
  // TODO: `:?` form bindings are temporary; see the TODO in
  // `HandleBindingPattern`.
  if (is_form || state.has_error) {
    return resolved_generic;
  }

  // Diagnose a phase keyword that is redundant with the contextual default. A
  // keyword that is invalid (rather than redundant) is left for `check`.
  if (generic_token && state.binding_context != BindingContext::ExplicitParam) {
    CARBON_DIAGNOSTIC(
        RedundantGenericModifier, Error,
        "`generic` is redundant here; this binding is a checked generic by "
        "default");
    context.emitter().Emit(*generic_token, RedundantGenericModifier);
    redundant_modifier = true;
  } else if (runtime_token &&
             state.binding_context == BindingContext::ExplicitParam) {
    CARBON_DIAGNOSTIC(
        RedundantRuntimeModifier, Error,
        "`runtime` is redundant here; this binding is runtime by default");
    context.emitter().Emit(*runtime_token, RedundantRuntimeModifier);
    redundant_modifier = true;
  }

  return resolved_generic;
}

auto HandleBindingPattern(Context& context) -> void {
  auto state = context.PopState();

  // Handle an invalid pattern introducer for parameters and variables.
  auto on_error = [&](bool expected_name, bool recover_as_raw = false) {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(
          ExpectedBindingPattern, Error,
          "expected {0:name|`:` or `:?`} in binding pattern"
          "{1:; prefix reserved word with `r#` to form a valid identifier|}",
          Diagnostics::BoolAsSelect, Diagnostics::BoolAsSelect);
      context.emitter().Emit(*context.position(), ExpectedBindingPattern,
                             expected_name, recover_as_raw);
      state.has_error = !recover_as_raw;
    }
  };

  // Phase keywords and `ref` may precede the name.
  auto template_token = context.ConsumeIf(Lex::TokenKind::Template);
  auto generic_token = context.ConsumeIf(Lex::TokenKind::Generic);
  auto runtime_token = context.ConsumeIf(Lex::TokenKind::Runtime);
  auto ref_token = context.ConsumeIf(Lex::TokenKind::Ref);
  if (ref_token && state.in_var_pattern) {
    CARBON_DIAGNOSTIC(RefInsideVar, Error, "found `ref` inside `var` pattern");
    context.emitter().Emit(*ref_token, RefInsideVar);
    state.has_error = true;
  }

  // Recover from `unused` written after a phase keyword or `ref` by consuming
  // it and wrapping the binding in `unused`, as if it had been written first.
  // The misordering is diagnosed later, once we know the modifier is itself
  // valid; a redundant or invalid modifier is diagnosed on its own, and we
  // don't stack the ordering error on top of it.
  std::optional<Lex::TokenIndex> misordered_unused_token;
  Lex::TokenKind misordered_unused_modifier = Lex::TokenKind::Unused;
  if ((template_token || generic_token || runtime_token || ref_token) &&
      context.PositionIs(Lex::TokenKind::Unused)) {
    misordered_unused_modifier = template_token  ? Lex::TokenKind::Template
                                 : generic_token ? Lex::TokenKind::Generic
                                 : runtime_token ? Lex::TokenKind::Runtime
                                                 : Lex::TokenKind::Ref;
    context.PushState(StateKind::FinishUnusedPattern);
    misordered_unused_token = context.ConsumeChecked(Lex::TokenKind::Unused);
  }

  // The first item should be an identifier, the placeholder `_`, or `self`.
  std::optional<Lex::TokenIndex> self_token;
  if (auto identifier = context.ConsumeIf(Lex::TokenKind::Identifier)) {
    context.AddLeafNode(NodeKind::IdentifierNameNotBeforeSignature,
                        *identifier);
  } else if (auto self =
                 context.ConsumeIf(Lex::TokenKind::SelfValueIdentifier)) {
    // Checking will validate where `self` may be declared. Its type may be
    // omitted, in which case it defaults to `Self` (see below).
    self_token = *self;
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

  auto token_kind = context.PositionKind();
  if (!token_kind.is_binding_pattern_operator()) {
    if (self_token && !template_token && !generic_token && !runtime_token) {
      // A `self` binding may omit its type; checking supplies the implicit
      // `Self` type. There is no type node, so this produces a
      // `SelfBindingPattern` rather than a `LetBindingPattern`.
      if (ref_token) {
        context.AddNode(NodeKind::RefBindingName, *ref_token, state.has_error);
      }
      context.AddNode(NodeKind::SelfBindingPattern, *self_token,
                      state.has_error);
      if (state.has_error) {
        context.ReturnErrorOnState();
      }
      return;
    }
    on_error(/*expected_name=*/false);
    // Add a substitute for the identifier name and virtual type-start nodes.
    context.AddInvalidParse(*context.position());
    context.AddInvalidParse(*context.position());
    context.PushState(state, StateKind::BindingPatternFinishAsRegular);
    return;
  }

  // TODO: `:?` introduces a form binding, from pending proposal #5389; proposal
  // #7254 suggests replacing it with a `fwd` binding modifier. Until then form
  // bindings are handled inline here, and are unaffected by phase keywords and
  // contextual defaults.
  bool is_form = token_kind == Lex::TokenKind::ColonQuestion;

  bool redundant_modifier = false;
  bool resolved_generic =
      ResolveBindingPhase(context, state, is_form, template_token,
                          generic_token, runtime_token, redundant_modifier);

  // `self` is always a runtime receiver binding; its phase never comes from the
  // enclosing context's default. Forcing runtime here means that a misplaced
  // `self` (in a deduced `[]` list or a compile-time entity's parameters, where
  // the default would otherwise be generic) is reported by `check` as a
  // misplaced `self` — the relevant error — rather than also producing a
  // `ref`-on-generic error from that default.
  if (self_token) {
    resolved_generic = false;
  }

  // `template` and `ref` wrap the binding name, and each is only meaningful on
  // a particular kind of binding: `template` on a generic binding, and `ref` on
  // a runtime `:` binding. Using one elsewhere is diagnosed and marks the
  // binding as errored; we skip its wrapper node rather than attach it to a
  // binding that can't hold it, which would leave the parse tree malformed.
  if (template_token) {
    if (is_form || !resolved_generic) {
      if (!state.has_error) {
        CARBON_DIAGNOSTIC(ExpectedGenericBindingPatternAfterTemplate, Error,
                          "`template` is only allowed on a generic binding");
        context.emitter().Emit(*template_token,
                               ExpectedGenericBindingPatternAfterTemplate);
      }
      state.has_error = true;
    } else {
      context.AddNode(NodeKind::TemplateBindingName, *template_token,
                      state.has_error);
    }
  }
  if (ref_token) {
    if (is_form || resolved_generic) {
      if (!state.has_error) {
        CARBON_DIAGNOSTIC(ExpectedRuntimeBindingPatternAfterRef, Error,
                          "`ref` is only allowed on a runtime binding");
        context.emitter().Emit(*ref_token,
                               ExpectedRuntimeBindingPatternAfterRef);
      }
      state.has_error = true;
    } else {
      context.AddNode(NodeKind::RefBindingName, *ref_token, state.has_error);
    }
  }
  // Preserve an explicit `runtime` keyword as a node wrapping the binding name,
  // so its token is accounted for and `check` sees that the phase was written.
  // It applies only to a runtime `:` binding; on a generic or form binding the
  // keyword doesn't set the phase and any misuse is diagnosed separately, so no
  // node is added there.
  if (runtime_token && !is_form && !resolved_generic) {
    context.AddNode(NodeKind::RuntimeBindingName, *runtime_token,
                    state.has_error);
  }

  // Now diagnose a misordered `unused` (recovered above), but only for an
  // otherwise-valid modifier: a redundant modifier sets `redundant_modifier`,
  // and an invalid `template`/`ref` sets `has_error`. Mark the binding in error
  // once diagnosed, since recovery reordered the tokens the user wrote.
  if (misordered_unused_token && !redundant_modifier && !state.has_error) {
    CARBON_DIAGNOSTIC(UnusedAfterBindingModifier, Error,
                      "`unused` must be written before `{0}`", Lex::TokenKind);
    context.emitter().Emit(*misordered_unused_token, UnusedAfterBindingModifier,
                           misordered_unused_modifier);
    state.has_error = true;
  }

  if (is_form) {
    state.kind = StateKind::BindingPatternFinishAsForm;
  } else if (resolved_generic) {
    state.kind = StateKind::BindingPatternFinishAsGeneric;
  } else {
    state.kind = StateKind::BindingPatternFinishAsRegular;
  }

  // Use the `:` or `:?` for the root node.
  state.token = context.Consume();

  // Add a virtual node before the binding's type expression.
  if (!is_form && resolved_generic) {
    context.AddLeafNode(NodeKind::CompileTimeBindingPatternTypeStart,
                        state.token, state.has_error);
  } else {
    context.AddLeafNode(NodeKind::BindingPatternTypeStart, state.token,
                        state.has_error);
  }

  context.PushState(state);
  context.PushStateForExpr(PrecedenceGroup::ForType());
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
                        "found {0:generic|`:?`} binding inside `var` pattern",
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
