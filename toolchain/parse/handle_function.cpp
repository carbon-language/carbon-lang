// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

auto HandleFunctionIntroducer(Context& context) -> void {
  auto state = context.PopState();
  context.PushState(state, StateKind::FunctionAfterParams);
  context.PushState(StateKind::DeclNameAndParams, state.token);
}

auto HandleFunctionAfterParams(Context& context) -> void {
  auto state = context.PopState();

  // Regardless of whether there's a return type, we'll finish the signature.
  context.PushState(state, StateKind::FunctionSignatureFinish);

  // If there is a return type, parse the expression before adding the return
  // type node.
  if (context.PositionIs(Lex::TokenKind::MinusGreater)) {
    context.PushState(StateKind::FunctionReturnTypeFinish);
    context.ConsumeAndDiscard();
    context.PushStateForExpr(PrecedenceGroup::ForType());
  }
}

auto HandleFunctionReturnTypeFinish(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::ReturnType, state.token, state.has_error);
}

auto HandleFunctionSignatureFinish(Context& context) -> void {
  auto state = context.PopState();

  switch (context.PositionKind()) {
    case Lex::TokenKind::Semi: {
      context.AddNode(NodeKind::FunctionDecl, context.Consume(),
                      state.has_error);
      break;
    }
    case Lex::TokenKind::OpenCurlyBrace: {
      context.AddFunctionDefinitionStart(context.Consume(), state.has_error);
      // Any error is recorded on the FunctionDefinitionStart.
      state.has_error = false;
      context.PushState(state, StateKind::FunctionDefinitionFinish);
      context.PushState(StateKind::StatementScopeLoop);
      break;
    }
    case Lex::TokenKind::Equal: {
      context.AddNode(NodeKind::BuiltinFunctionDefinitionStart,
                      context.Consume(), state.has_error);
      if (!context.ConsumeAndAddLeafNodeIf(Lex::TokenKind::StringLiteral,
                                           NodeKind::BuiltinName)) {
        CARBON_DIAGNOSTIC(ExpectedBuiltinName, Error,
                          "expected builtin function name after `=`");
        context.emitter().Emit(*context.position(), ExpectedBuiltinName);
        state.has_error = true;
      }
      auto semi = context.ConsumeIf(Lex::TokenKind::Semi);
      if (!semi && !state.has_error) {
        context.DiagnoseExpectedDeclSemi(context.tokens().GetKind(state.token));
        state.has_error = true;
      }
      if (state.has_error) {
        context.RecoverFromDeclError(state, NodeKind::BuiltinFunctionDefinition,
                                     /*skip_past_likely_end=*/true);
      } else {
        context.AddNode(NodeKind::BuiltinFunctionDefinition, *semi,
                        state.has_error);
      }
      break;
    }
    default: {
      if (!state.has_error) {
        context.DiagnoseExpectedDeclSemiOrDefinition(Lex::TokenKind::Fn);
      }
      // Only need to skip if we've not already found a new line.
      bool skip_past_likely_end =
          context.tokens().GetLine(*context.position()) ==
          context.tokens().GetLine(state.token);
      context.RecoverFromDeclError(state, NodeKind::FunctionDecl,
                                   skip_past_likely_end);
      break;
    }
  }
}

auto HandleFunctionDefinitionFinish(Context& context) -> void {
  auto state = context.PopState();
  context.AddFunctionDefinition(context.Consume(), state.has_error);
}

}  // namespace Carbon::Parse
