// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

// Handles processing of a complete `adapt T` declaration.
auto HandleAdaptDecl(Context& context) -> void {
  // TODO: This is identical to HandleBaseDecl other than the `NodeKind`,
  // and very similar to `HandleNamespaceFinish` and `HandleAliasFinish`.
  // We should factor out this common work.
  auto state = context.PopState();

  auto semi = context.ConsumeIf(Lex::TokenKind::Semi);
  if (!semi && !state.has_error) {
    context.DiagnoseExpectedDeclSemi(context.tokens().GetKind(state.token));
    state.has_error = true;
  }

  if (state.has_error) {
    context.RecoverFromDeclError(state, NodeKind::AdaptDecl,
                                 /*skip_past_likely_end=*/true);
    return;
  }

  context.AddNode(NodeKind::AdaptDecl, *semi, state.subtree_start,
                  state.has_error);
}

}  // namespace Carbon::Parse
