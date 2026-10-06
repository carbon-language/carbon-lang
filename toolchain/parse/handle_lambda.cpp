// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

auto HandleLambdaIntroducer(Context& context) -> void {
  auto state = context.PopState();
  context.AddLeafNode(NodeKind::LambdaIntroducer, context.Consume());
  context.PushState(state, StateKind::LambdaAfterIntroducer);
}

auto HandleLambdaAfterIntroducer(Context& context) -> void {
  auto state = context.PopState();

  if (context.PositionIs(Lex::TokenKind::OpenSquareBracket)) {
    context.PushState(state, StateKind::LambdaAfterImplicitParams);
    context.PushState(StateKind::PatternListAsImplicit, *context.position(),
                      BindingContext::DeducedParam);
  } else if (context.PositionIs(Lex::TokenKind::OpenParen)) {
    context.PushState(state, StateKind::LambdaAfterParams);
    context.PushState(StateKind::PatternListAsExplicit);
  } else {
    // No implicit or explicit params.
    context.PushState(state, StateKind::LambdaAfterParams);
  }
}

auto HandleLambdaAfterImplicitParams(Context& context) -> void {
  auto state = context.PopState();

  if (context.PositionIs(Lex::TokenKind::OpenParen)) {
    context.PushState(state, StateKind::LambdaAfterParams);
    context.PushState(StateKind::PatternListAsExplicit);
  } else {
    // No explicit params after implicit params.
    context.PushState(state, StateKind::LambdaAfterParams);
  }
}

static auto ParseLambdaBody(Context& context, Context::State state,
                            bool has_return_type) -> void {
  if (context.PositionIs(Lex::TokenKind::EqualGreater)) {
    // Terse body `=> expr`
    auto arrow_token = context.Consume();
    context.AddNode(NodeKind::LambdaDefinitionStart, arrow_token,
                    state.has_error);
    state.has_error = false;
    context.AddLeafNode(NodeKind::TerseBodyArrow, arrow_token);
    context.PushState(state, StateKind::LambdaBodyFinish);
    context.PushStateForExpr(PrecedenceGroup::ForTopLevelExpr());
  } else if (context.PositionIs(Lex::TokenKind::OpenCurlyBrace)) {
    // Block body `{ ... }`
    context.PushState(StateKind::LambdaBodyFinish);
    context.AddNode(NodeKind::LambdaDefinitionStart, context.Consume(),
                    state.has_error);
    context.PushState(StateKind::StatementScopeLoop);
  } else {
    if (has_return_type) {
      CARBON_DIAGNOSTIC(ExpectedLambdaBodyAfterReturnType, Error,
                        "expected `=>` or `{{` after return type");
      context.emitter().Emit(*context.position(),
                             ExpectedLambdaBodyAfterReturnType);
    } else {
      CARBON_DIAGNOSTIC(ExpectedLambdaBody, Error,
                        "expected `->`, `=>`, or `{{`");
      context.emitter().Emit(*context.position(), ExpectedLambdaBody);
    }

    // Bundle everything into a complete lambda node for error recovery --
    // otherwise the orphaned `LambdaIntroducer` would be left where an
    // expression is required, for example in `(fn)`.
    context.AddNode(NodeKind::LambdaDefinitionStart, *context.position(),
                    /*has_error=*/true);
    context.AddNode(NodeKind::Lambda, state.token, /*has_error=*/true);
  }
}

auto HandleLambdaAfterParams(Context& context) -> void {
  auto state = context.PopState();

  if (context.PositionIs(Lex::TokenKind::MinusGreater)) {
    // Has return type.
    context.PushState(state, StateKind::LambdaBody);
    context.PushState(StateKind::FunctionReturnTypeFinish);
    context.ConsumeAndDiscard();
    context.PushStateForExpr(PrecedenceGroup::ForType());
  } else if (context.PositionIs(Lex::TokenKind::MinusGreaterQuestion)) {
    // Has return form.
    context.PushState(state, StateKind::LambdaBody);
    context.PushState(StateKind::FunctionReturnFormFinish);
    context.ConsumeAndDiscard();
    context.PushStateForExpr(PrecedenceGroup::ForType());
  } else {
    ParseLambdaBody(context, state, /*has_return_type=*/false);
  }
}

auto HandleLambdaBody(Context& context) -> void {
  auto state = context.PopState();
  ParseLambdaBody(context, state, /*has_return_type=*/true);
}

auto HandleLambdaBodyFinish(Context& context) -> void {
  auto state = context.PopState();
  if (context.tokens().GetKind(state.token) == Lex::TokenKind::OpenCurlyBrace) {
    context.AddNode(NodeKind::Lambda, context.Consume(), state.has_error);
  } else {
    context.AddNode(NodeKind::Lambda, state.token, state.has_error);
  }
}

}  // namespace Carbon::Parse
