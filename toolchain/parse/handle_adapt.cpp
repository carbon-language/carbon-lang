// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

// Handles processing of a complete `adapt T` declaration.
auto HandleAdaptAfterIntroducer(Context& context) -> void {
  auto state = context.PopState();
  state.kind = StateKind::AdaptDecl;
  context.PushState(state);
  context.PushState(StateKind::Expr);
}

// Handles processing of a complete `adapt T` declaration.
auto HandleAdaptDecl(Context& context) -> void {
  // TODO: This is identical to HandleBaseDecl other than the `NodeKind`,
  // and very similar to `HandleNamespaceFinish` and `HandleAliasFinish`.
  // We should factor out this common work.
  auto state = context.PopState();

  context.AddNodeExpectingDeclSemi(state, NodeKind::AdaptDecl,
                                   Lex::TokenKind::Adapt,
                                   /*is_def_allowed=*/false);
}

}  // namespace Carbon::Parse
