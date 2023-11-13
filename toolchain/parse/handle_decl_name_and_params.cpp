// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

// Handles DeclNameAndParamsAs(Optional|Required).
static auto HandleDeclNameAndParams(Context& context, State after_name)
    -> void {
  auto state = context.PopState();

  // TODO: Should handle designated names.
  if (auto identifier = context.ConsumeIf(Lex::TokenKind::Identifier)) {
    state.state = after_name;
    context.PushState(state);

    if (context.PositionIs(Lex::TokenKind::Period)) {
      context.AddLeafNode(NodeKind::Name, *identifier);
      state.state = State::PeriodAsDecl;
      context.PushState(state);
    } else {
      context.AddLeafNode(NodeKind::Name, *identifier);
    }
  } else {
    CARBON_DIAGNOSTIC(ExpectedDeclName, Error,
                      "`{0}` introducer should be followed by a name.",
                      Lex::TokenKind);
    context.emitter().Emit(*context.position(), ExpectedDeclName,
                           context.tokens().GetKind(state.token));
    context.ReturnErrorOnState();
    context.AddLeafNode(NodeKind::InvalidParse, *context.position(),
                        /*has_error=*/true);
  }
}

auto HandleDeclNameAndParamsAsNone(Context& context) -> void {
  HandleDeclNameAndParams(context, State::DeclNameAndParamsAfterNameAsNone);
}

auto HandleDeclNameAndParamsAsOptional(Context& context) -> void {
  HandleDeclNameAndParams(context, State::DeclNameAndParamsAfterNameAsOptional);
}

auto HandleDeclNameAndParamsAsRequired(Context& context) -> void {
  HandleDeclNameAndParams(context, State::DeclNameAndParamsAfterNameAsRequired);
}

enum class Params : int8_t {
  None,
  Optional,
  Required,
};

static auto HandleDeclNameAndParamsAfterName(Context& context, Params params)
    -> void {
  auto state = context.PopState();

  if (context.PositionIs(Lex::TokenKind::Period)) {
    // Continue designator processing.
    context.PushState(state);
    state.state = State::PeriodAsDecl;
    context.PushState(state);
    return;
  }

  // TODO: We can have a parameter list after a name qualifier, regardless of
  // whether the entity itself permits or requires parameters:
  //
  //   fn Class(T:! type).AnotherClass(U:! type).Function(v: T) {}
  //
  // We should retain a `DeclNameAndParams...` state on the stack in all
  // cases below to check for a period after a parameter list, which indicates
  // that we've not finished parsing the declaration name.

  if (params == Params::None) {
    return;
  }

  if (context.PositionIs(Lex::TokenKind::OpenSquareBracket)) {
    context.PushState(State::DeclNameAndParamsAfterImplicit);
    context.PushState(State::ParamListAsImplicit);
  } else if (context.PositionIs(Lex::TokenKind::OpenParen)) {
    context.PushState(State::ParamListAsRegular);
  } else if (params == Params::Required) {
    CARBON_DIAGNOSTIC(ParamsRequiredByIntroducer, Error,
                      "`{0}` requires a `(` for parameters.", Lex::TokenKind);
    context.emitter().Emit(*context.position(), ParamsRequiredByIntroducer,
                           context.tokens().GetKind(state.token));
    context.ReturnErrorOnState();
  }
}

auto HandleDeclNameAndParamsAfterNameAsNone(Context& context) -> void {
  HandleDeclNameAndParamsAfterName(context, Params::None);
}

auto HandleDeclNameAndParamsAfterNameAsOptional(Context& context) -> void {
  HandleDeclNameAndParamsAfterName(context, Params::Optional);
}

auto HandleDeclNameAndParamsAfterNameAsRequired(Context& context) -> void {
  HandleDeclNameAndParamsAfterName(context, Params::Required);
}

auto HandleDeclNameAndParamsAfterImplicit(Context& context) -> void {
  context.PopAndDiscardState();

  if (context.PositionIs(Lex::TokenKind::OpenParen)) {
    context.PushState(State::ParamListAsRegular);
  } else {
    CARBON_DIAGNOSTIC(
        ParamsRequiredAfterImplicit, Error,
        "A `(` for parameters is required after implicit parameters.");
    context.emitter().Emit(*context.position(), ParamsRequiredAfterImplicit);
    context.ReturnErrorOnState();
  }
}

}  // namespace Carbon::Parse
