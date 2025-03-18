// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/token_index.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

// Adds a leaf node for the name, and updates the state stack for parameter
// handling.
static auto HandleName(Context& context, Context::StateStackEntry state,
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
      context.PushState(State::DeclNameAndParams);
      break;

    case Lex::TokenKind::OpenSquareBracket:
      context.AddLeafNode(before_params_kind, name_token);
      state.state = State::DeclNameAndParamsAfterImplicit;
      context.PushState(state);
      context.PushState(State::PatternListAsImplicit);
      break;

    case Lex::TokenKind::OpenParen:
      context.AddLeafNode(before_params_kind, name_token);
      state.state = State::DeclNameAndParamsAfterParams;
      context.PushState(state);
      context.PushState(State::PatternListAsExplicit);
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
               NodeKind::IdentifierNameNotBeforeParams,
               NodeKind::IdentifierNameQualifierWithoutParams,
               NodeKind::IdentifierNameBeforeParams);
    return;
  }

  if (auto keyword = context.ConsumeIf(Lex::TokenKind::Destroy)) {
    HandleName(context, state, *keyword, NodeKind::KeywordNameNotBeforeParams,
               NodeKind::KeywordNameQualifierWithoutParams,
               NodeKind::KeywordNameBeforeParams);
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

  state.state = State::DeclNameAndParamsAfterParams;
  context.PushState(state);

  if (!context.PositionIs(Lex::TokenKind::OpenParen)) {
    return;
  }

  context.PushState(State::PatternListAsExplicit);
}

auto HandleDeclNameAndParamsAfterParams(Context& context) -> void {
  auto state = context.PopState();

  if (auto period = context.ConsumeIf(Lex::TokenKind::Period)) {
    auto start_kind = context.tree().node_kind(NodeId(state.subtree_start));
    auto node_kind = start_kind == NodeKind::IdentifierNameBeforeParams
                         ? NodeKind::IdentifierNameQualifierWithParams
                         : NodeKind::KeywordNameQualifierWithParams;
    context.AddNode(node_kind, *period, state.has_error);
    context.PushState(State::DeclNameAndParams);
  }
}

}  // namespace Carbon::Parse
