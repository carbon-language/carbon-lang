// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

namespace {
enum class Variant { NoParams, QualifierParams, AllParams };
}  // namespace

static auto BaseState(Variant variant) -> State {
  switch (variant) {
    case Variant::NoParams:
      return State::DeclNameAndParamsAsNoParams;
    case Variant::QualifierParams:
      return State::DeclNameAndParamsAsQualifierParams;
    case Variant::AllParams:
      return State::DeclNameAndParamsAsAllParams;
  }
}

static auto AfterImplicitState(Variant variant) {
  switch (variant) {
    case Variant::NoParams:
      CARBON_FATAL("State does not exist");
    case Variant::QualifierParams:
      return State::DeclNameAndParamsAfterImplicitAsQualifierParams;
    case Variant::AllParams:
      return State::DeclNameAndParamsAfterImplicitAsAllParams;
  }
}

static auto AfterParamsState(Variant variant) {
  switch (variant) {
    case Variant::NoParams:
      CARBON_FATAL("State does not exist");
    case Variant::QualifierParams:
      return State::DeclNameAndParamsAfterParamsAsQualifierParams;
    case Variant::AllParams:
      return State::DeclNameAndParamsAfterParamsAsAllParams;
  }
}

static auto HandleDeclNameAndParams(Context& context, Variant variant) -> void {
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

  auto diagnose_params = [&] {
    CARBON_DIAGNOSTIC(UnexpectedParamsInDeclName, Error,
                      "unexpected parameters in name declaration");
    context.emitter().Emit(state.token, UnexpectedParamsInDeclName);
    context.ReturnErrorOnState();
  };

  switch (context.PositionKind()) {
    case Lex::TokenKind::Period: {
      context.AddLeafNode(NodeKind::IdentifierNameNotBeforeParams, *identifier);
      context.AddNode(NodeKind::NameQualifierWithoutParams,
                      context.ConsumeChecked(Lex::TokenKind::Period),
                      state.has_error);
      context.PushState(BaseState(variant));

      break;
    }
    case Lex::TokenKind::OpenSquareBracket: {
      if (variant == Variant::NoParams) {
        diagnose_params();
        return;
      }
      context.AddLeafNode(NodeKind::IdentifierNameBeforeParams, *identifier);
      state.state = AfterImplicitState(variant);
      context.PushState(state);
      context.PushState(State::PatternListAsImplicit);
      break;
    }
    case Lex::TokenKind::OpenParen: {
      if (variant == Variant::NoParams) {
        diagnose_params();
        return;
      }
      context.AddLeafNode(NodeKind::IdentifierNameBeforeParams, *identifier);
      state.state = AfterParamsState(variant);
      context.PushState(state);
      context.PushState(State::PatternListAsTuple);
      break;
    }

    default:
      context.AddLeafNode(NodeKind::IdentifierNameNotBeforeParams, *identifier);
      break;
  }
}

auto HandleDeclNameAndParamsAsNoParams(Context& context) -> void {
  HandleDeclNameAndParams(context, Variant::NoParams);
}

auto HandleDeclNameAndParamsAsQualifierParams(Context& context) -> void {
  HandleDeclNameAndParams(context, Variant::QualifierParams);
}

auto HandleDeclNameAndParamsAsAllParams(Context& context) -> void {
  HandleDeclNameAndParams(context, Variant::AllParams);
}

static auto HandleDeclNameAndParamsAfterImplicit(Context& context,
                                                 Variant variant) -> void {
  auto state = context.PopState();

  if (!context.PositionIs(Lex::TokenKind::OpenParen)) {
    CARBON_DIAGNOSTIC(
        ParamsRequiredAfterImplicit, Error,
        "a `(` for parameters is required after implicit parameters");
    context.emitter().Emit(*context.position(), ParamsRequiredAfterImplicit);
    context.ReturnErrorOnState();
    return;
  }

  state.state = AfterParamsState(variant);
  context.PushState(state);
  context.PushState(State::PatternListAsTuple);
}

auto HandleDeclNameAndParamsAfterImplicitAsQualifierParams(Context& context)
    -> void {
  HandleDeclNameAndParamsAfterImplicit(context, Variant::QualifierParams);
}

auto HandleDeclNameAndParamsAfterImplicitAsAllParams(Context& context) -> void {
  HandleDeclNameAndParamsAfterImplicit(context, Variant::AllParams);
}

static auto HandleDeclNameAndParamsAfterParams(Context& context,
                                               Variant variant) -> void {
  auto state = context.PopState();

  if (auto period = context.ConsumeIf(Lex::TokenKind::Period)) {
    context.AddNode(NodeKind::NameQualifierWithParams, *period,
                    state.has_error);
    context.PushState(BaseState(variant));
  } else {
    if (variant != Variant::AllParams) {
      CARBON_DIAGNOSTIC(UnexpectedParamsAfterDeclName, Error,
                        "unexpected parameters after name declaration");
      context.emitter().Emit(*context.position(),
                             UnexpectedParamsAfterDeclName);
      context.ReturnErrorOnState();
      return;
    }
  }
}

auto HandleDeclNameAndParamsAfterParamsAsQualifierParams(Context& context)
    -> void {
  HandleDeclNameAndParamsAfterParams(context, Variant::QualifierParams);
}

auto HandleDeclNameAndParamsAfterParamsAsAllParams(Context& context) -> void {
  HandleDeclNameAndParamsAfterParams(context, Variant::AllParams);
}

}  // namespace Carbon::Parse
