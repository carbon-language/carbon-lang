// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/token_index.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

// Determines the binding context for an explicit `()` parameter list beginning
// at the current position. A list that is a name qualifier (its matched `)` is
// immediately followed by `.`) names a compile-time entity, so its parameters
// use the compile-time-entity context; otherwise the list uses the enclosing
// declaration's default `decl_context`.
static auto ExplicitParamListContext(Context& context,
                                     BindingContext decl_context)
    -> BindingContext {
  // A compile-time entity's parameters are compile-time whether or not this
  // list is a name qualifier, so there is no need to look ahead for the `.`.
  if (decl_context == BindingContext::CompileTimeEntityParam) {
    return decl_context;
  }
  auto open_paren = *context.position();
  auto close_paren = context.tokens().GetMatchedClosingToken(open_paren);
  auto after_close = Lex::TokenIndex(close_paren.index + 1);
  if (context.tokens().GetKind(after_close) == Lex::TokenKind::Period) {
    return BindingContext::CompileTimeEntityParam;
  }
  return decl_context;
}

// Adds a leaf node for the name, and updates the state stack for parameter
// handling.
static auto HandleName(Context& context, Context::State state,
                       Lex::TokenIndex name_token,
                       NodeKind not_before_params_kind,
                       NodeKind not_before_params_qualifier_kind,
                       NodeKind before_params_kind) -> void {
  switch (context.PositionKind()) {
    case Lex::TokenKind::Period:
      context.AddLeafNode(not_before_params_kind, name_token);
      context.AddNode(not_before_params_qualifier_kind,
                      context.ConsumeChecked(Lex::TokenKind::Period),
                      state.has_error);
      // Carry the declaration's binding context across the qualified name so
      // the final parameter list uses the declaration's default phase.
      context.PushState(StateKind::DeclNameAndParams, *context.position(),
                        state.binding_context);
      break;

    case Lex::TokenKind::OpenSquareBracket:
      context.AddLeafNode(before_params_kind, name_token);
      state.kind = StateKind::DeclNameAndParamsAfterImplicit;
      context.PushState(state);
      context.PushState(StateKind::PatternListAsImplicit, *context.position(),
                        BindingContext::DeducedParam);
      break;

    case Lex::TokenKind::OpenParen:
      context.AddLeafNode(before_params_kind, name_token);
      state.kind = StateKind::DeclNameAndParamsAfterParams;
      context.PushState(state);
      context.PushState(
          StateKind::PatternListAsExplicit, *context.position(),
          ExplicitParamListContext(context, state.binding_context));
      break;

    case Lex::TokenKind::MinusGreater:
    case Lex::TokenKind::MinusGreaterQuestion:
      context.AddLeafNode(before_params_kind, name_token);
      break;

    default:
      context.AddLeafNode(not_before_params_kind, name_token);
      break;
  }
}

auto HandleDeclNameAndParams(Context& context) -> void {
  auto state = context.PopState();

  if (auto identifier = context.ConsumeIf(Lex::TokenKind::Identifier)) {
    HandleName(context, state, *identifier,
               NodeKind::IdentifierNameNotBeforeSignature,
               NodeKind::IdentifierNameQualifierWithoutParams,
               NodeKind::IdentifierNameMaybeBeforeSignature);
    return;
  }

  Lex::TokenIndex token = *context.position();
  if (context.tokens().GetKind(token) == Lex::TokenKind::FileEnd) {
    // The end of file is an unhelpful diagnostic location. Instead, use the
    // introducer token.
    token = state.token;
  }
  if (state.token == *context.position()) {
    CARBON_DIAGNOSTIC(ExpectedDeclNameAfterPeriod, Error,
                      "`.` should be followed by a name");
    context.emitter().Emit(token, ExpectedDeclNameAfterPeriod);
  } else {
    CARBON_DIAGNOSTIC(ExpectedDeclName, Error,
                      "`{0}` introducer should be followed by a name",
                      Lex::TokenKind);
    context.emitter().Emit(token, ExpectedDeclName,
                           context.tokens().GetKind(state.token));
  }
  context.ReturnErrorOnState();
  context.AddInvalidParse(*context.position());
}

auto HandleDeclNameAndParamsAfterImplicit(Context& context) -> void {
  auto state = context.PopState();

  state.kind = StateKind::DeclNameAndParamsAfterParams;
  context.PushState(state);

  if (!context.PositionIs(Lex::TokenKind::OpenParen)) {
    return;
  }

  context.PushState(StateKind::PatternListAsExplicit, *context.position(),
                    ExplicitParamListContext(context, state.binding_context));
}

auto HandleDeclNameAndParamsAfterParams(Context& context) -> void {
  auto state = context.PopState();

  if (auto period = context.ConsumeIf(Lex::TokenKind::Period)) {
    CARBON_CHECK(context.tree().node_kind(NodeId(state.subtree_start)) ==
                 NodeKind::IdentifierNameMaybeBeforeSignature);
    context.AddNode(NodeKind::IdentifierNameQualifierWithParams, *period,
                    state.has_error);
    context.PushState(StateKind::DeclNameAndParams, *context.position(),
                      state.binding_context);
  }
}

}  // namespace Carbon::Parse
