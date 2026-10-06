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
    context.PushState(StateKind::PatternListAsImplicit);
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

auto HandleLambdaAfterParams(Context& context) -> void {
  auto state = context.PopState();

  if (context.PositionIs(Lex::TokenKind::MinusGreater)) {
    // Has return type.
    context.PushState(state, StateKind::LambdaBody);
    context.PushState(StateKind::FunctionReturnTypeFinish);
    context.ConsumeAndDiscard();
    context.PushStateForExpr(PrecedenceGroup::ForType());
  } else if (context.PositionIs(Lex::TokenKind::EqualGreater)) {
    // Terse body `=> expr`
    context.AddLeafNode(NodeKind::TerseBodyArrow, context.Consume());
    context.PushState(state, StateKind::LambdaBodyFinish);
    context.PushStateForExpr(PrecedenceGroup::ForTopLevelExpr());
  } else if (context.PositionIs(Lex::TokenKind::OpenCurlyBrace)) {
    // Block body `{ ... }`
    context.PushState(state, StateKind::LambdaBodyFinish);
    context.PushState(StateKind::CodeBlock);
  } else {
    CARBON_DIAGNOSTIC(ExpectedLambdaBody, Error,
                      "expected `->`, `=>`, or `{{`");
    context.emitter().Emit(*context.position(), ExpectedLambdaBody);

    // Add a dummy node for the missing body without consuming the current
    // token, then bundle everything into a complete lambda node. This keeps the
    // lambda a valid expression for error recovery -- otherwise the orphaned
    // `LambdaIntroducer` would be left where an expression is required, for
    // example in `(fn)`.
    context.AddLeafNode(NodeKind::InvalidParse, *context.position(),
                        /*has_error=*/true);

    state.has_error = true;
    context.PushState(state, StateKind::LambdaBodyFinish);
  }
}

auto HandleLambdaBody(Context& context) -> void {
  auto state = context.PopState();

  // We arrive here after parsing return type.
  // So we look for `=>` or `{`.

  if (context.PositionIs(Lex::TokenKind::EqualGreater)) {
    // Terse body `=> expr`
    context.AddLeafNode(NodeKind::TerseBodyArrow, context.Consume());
    context.PushState(state, StateKind::LambdaBodyFinish);
    context.PushStateForExpr(PrecedenceGroup::ForTopLevelExpr());
  } else if (context.PositionIs(Lex::TokenKind::OpenCurlyBrace)) {
    // Block body `{ ... }`
    context.PushState(state, StateKind::LambdaBodyFinish);
    context.PushState(StateKind::CodeBlock);
  } else {
    CARBON_DIAGNOSTIC(ExpectedLambdaBodyAfterReturnType, Error,
                      "expected `=>` or `{{` after return type");
    context.emitter().Emit(*context.position(),
                           ExpectedLambdaBodyAfterReturnType);

    // Add a dummy node for the missing body without consuming the current
    // token.
    context.AddLeafNode(NodeKind::InvalidParse, *context.position(),
                        /*has_error=*/true);

    state.has_error = true;
    // Bundle all nodes into a complete lambda node.
    context.PushState(state, StateKind::LambdaBodyFinish);
  }
}

auto HandleLambdaBodyFinish(Context& context) -> void {
  auto state = context.PopState();
  context.AddNode(NodeKind::Lambda, state.token, state.has_error);
}

}  // namespace Carbon::Parse
