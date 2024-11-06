// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

auto HandleDeclNameAndParams(Context& context) -> void {
  auto state = context.PopState();

  auto identifier = context.ConsumeIf(Lex::TokenKind::Identifier);
  if (!identifier) {
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
    return;
  }

  context.AddLeafNode(NodeKind::IdentifierName, *identifier);

  switch (context.PositionKind()) {
    case Lex::TokenKind::Period:
      context.AddNode(NodeKind::NameQualifier,
                      context.ConsumeChecked(Lex::TokenKind::Period),
                      state.has_error);
      context.PushState(State::DeclNameAndParams);
      break;

    case Lex::TokenKind::OpenSquareBracket:
      state.state = State::DeclNameAndParamsAfterImplicit;
      context.PushState(state);
      context.PushState(State::PatternListAsImplicit);
      break;

    case Lex::TokenKind::OpenParen:
      state.state = State::DeclNameAndParamsAfterParams;
      context.PushState(state);
      context.PushState(State::PatternListAsTuple);
      break;

    default:
      break;
  }
}

auto HandleDeclNameAndParamsAfterImplicit(Context& context) -> void {
  auto state = context.PopState();

  if (!context.PositionIs(Lex::TokenKind::OpenParen)) {
    CARBON_DIAGNOSTIC(
        ParamsRequiredAfterImplicit, Error,
        "a `(` for parameters is required after implicit parameters");
    context.emitter().Emit(*context.position(), ParamsRequiredAfterImplicit);
    context.ReturnErrorOnState();
    return;
  }

  state.state = State::DeclNameAndParamsAfterParams;
  context.PushState(state);
  context.PushState(State::PatternListAsTuple);
}

auto HandleDeclNameAndParamsAfterParams(Context& context) -> void {
  auto state = context.PopState();

  if (auto period = context.ConsumeIf(Lex::TokenKind::Period)) {
    context.AddNode(NodeKind::NameQualifier, *period, state.has_error);
    context.PushState(State::DeclNameAndParams);
  }
}

}  // namespace Carbon::Parse
