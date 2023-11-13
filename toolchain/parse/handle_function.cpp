// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

auto HandleFunctionIntroducer(Context& context) -> void {
  auto state = context.PopState();

  context.AddLeafNode(NodeKind::FunctionIntroducer, context.Consume());

  state.state = State::FunctionAfterParams;
  context.PushState(state);
  context.PushState(State::DeclNameAndParamsAsRequired, state.token);
}

auto HandleFunctionAfterParams(Context& context) -> void {
  auto state = context.PopState();

  // Regardless of whether there's a return type, we'll finish the signature.
  state.state = State::FunctionSignatureFinish;
  context.PushState(state);

  // If there is a return type, parse the expression before adding the return
  // type node.
  if (context.PositionIs(Lex::TokenKind::MinusGreater)) {
    context.PushState(State::FunctionReturnTypeFinish);
    context.ConsumeAndDiscard();
    context.PushStateForExpr(PrecedenceGroup::ForType());
  }
}

auto HandleFunctionReturnTypeFinish(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::ReturnType, state.token, state.subtree_start,
                  state.has_error);
}

auto HandleFunctionSignatureFinish(Context& context) -> void {
  auto state = context.PopState();

  switch (context.PositionKind()) {
    case Lex::TokenKind::Semi: {
      context.AddNode(NodeKind::FunctionDecl, context.Consume(),
                      state.subtree_start, state.has_error);
      break;
    }
    case Lex::TokenKind::OpenCurlyBrace: {
      if (auto decl_context = context.GetDeclContext();
          decl_context == Context::DeclContext::Interface ||
          decl_context == Context::DeclContext::NamedConstraint) {
        CARBON_DIAGNOSTIC(
            MethodImplNotAllowed, Error,
            "Method implementations are not allowed in interfaces.");
        context.emitter().Emit(*context.position(), MethodImplNotAllowed);
        context.RecoverFromDeclError(state, NodeKind::FunctionDecl,
                                     /*skip_past_likely_end=*/true);
        break;
      }

      context.AddNode(NodeKind::FunctionDefinitionStart, context.Consume(),
                      state.subtree_start, state.has_error);
      // Any error is recorded on the FunctionDefinitionStart.
      state.has_error = false;
      state.state = State::FunctionDefinitionFinish;
      context.PushState(state);
      context.PushState(State::StatementScopeLoop);
      break;
    }
    default: {
      if (!state.has_error) {
        context.EmitExpectedDeclSemiOrDefinition(Lex::TokenKind::Fn);
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
  context.AddNode(NodeKind::FunctionDefinition, context.Consume(),
                  state.subtree_start, state.has_error);
}

}  // namespace Carbon::Parse
