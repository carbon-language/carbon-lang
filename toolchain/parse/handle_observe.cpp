// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/parse/precedence.h"
#include "toolchain/parse/state.h"
#include "toolchain/parse/typed_nodes.h"

namespace Carbon::Parse {
auto HandleObserveAfterIntroducer(Context& context) -> void {
  auto state = context.PopState();

  context.PushState(state, StateKind::ObserveOperator);
  context.PushStateForExpr(PrecedenceGroup::ForRequirements());
}

auto HandleObserveOperator(Context& context) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    context.ReturnErrorOnState();
  }

  switch (context.PositionKind()) {
    case Lex::TokenKind::EqualEqual: {
      context.AddLeafNode(NodeKind::ObserveEqualEqual, context.Consume());
      context.PushState(state, StateKind::ObserveOperator);
      context.PushStateForExpr(PrecedenceGroup::ForRequirements());
      return;
    }
    case Lex::TokenKind::Impls: {
      context.AddLeafNode(NodeKind::ObserveImpls, context.Consume());
      context.PushState(state, StateKind::ObserveDecl);
      context.PushState(StateKind::Expr);
      return;
    }
    default: {
      if (!state.has_error) {
        CARBON_DIAGNOSTIC(ExpectedObserveOperator, Error,
                          "observe should use `==` or `impls` operator");
        context.emitter().Emit(*context.position(), ExpectedObserveOperator);
      }
      context.ReturnErrorOnState();
      return;
    }
  }
}

auto HandleObserveDecl(Context& context) -> void {
  auto state = context.PopState();
  context.AddNodeExpectingDeclSemi(state, NodeKind::ObserveDecl,
                                   Lex::TokenKind::Observe,
                                   /*is_def_allowed=*/false);
}
}  // namespace Carbon::Parse
