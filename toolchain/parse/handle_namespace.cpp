// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

auto HandleNamespace(Context& context) -> void {
  auto state = context.PopState();
  context.PushState(state, State::NamespaceFinish);
  context.PushState(State::DeclNameAndParamsAsNone, state.token);
}

auto HandleNamespaceFinish(Context& context) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    context.RecoverFromDeclError(state, NodeKind::Namespace,
                                 /*skip_past_likely_end=*/true);
    return;
  }

  if (auto semi = context.ConsumeIf(Lex::TokenKind::Semi)) {
    context.AddNode(NodeKind::Namespace, *semi, state.subtree_start,
                    state.has_error);
  } else {
    context.EmitExpectedDeclSemi(Lex::TokenKind::Namespace);
    context.RecoverFromDeclError(state, NodeKind::Namespace,
                                 /*skip_past_likely_end=*/true);
  }
}

}  // namespace Carbon::Parse
