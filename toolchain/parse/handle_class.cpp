// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

// Processing of the `class` declaration and definition themselves is shared
// with interfaces and constraints, and is located in handle_type.cpp.

// Handles processing of a complete `base: B` declaration.
auto HandleBaseDecl(Context& context) -> void {
  auto state = context.PopState();

  auto semi = context.ConsumeIf(Lex::TokenKind::Semi);
  if (!semi && !state.has_error) {
    context.EmitExpectedDeclSemi(context.tokens().GetKind(state.token));
    state.has_error = true;
  }

  if (state.has_error) {
    context.RecoverFromDeclError(state, NodeKind::BaseDecl,
                                 /*skip_past_likely_end=*/true);
    return;
  }

  context.AddNode(NodeKind::BaseDecl, *semi, state.subtree_start,
                  state.has_error);
}

}  // namespace Carbon::Parse
