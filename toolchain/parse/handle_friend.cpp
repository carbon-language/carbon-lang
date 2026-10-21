// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

// Handles processing of a complete `friend X` declaration.
auto HandleFriendDecl(Context& context) -> void {
  auto state = context.PopState();
  if (state.has_error) {
    context.RecoverFromDeclError(state, NodeKind::FriendDecl,
                                 /*skip_past_likely_end=*/true);
    return;
  }

  if (!context.ConsumeAndAddLeafNodeIf(Lex::TokenKind::Identifier,
                                       NodeKind::IdentifierNameExpr)) {
    state.has_error = true;
    CARBON_DIAGNOSTIC(ExpectedIdentifierForFriend, Error,
                      "expected identifier for friend declaration");
    context.emitter().Emit(*context.position(), ExpectedIdentifierForFriend);
    context.RecoverFromDeclError(state, NodeKind::FriendDecl,
                                 /*skip_past_likely_end=*/true);
    return;
  }
  context.AddNodeExpectingDeclSemi(state, NodeKind::FriendDecl,
                                   Lex::TokenKind::Friend,
                                   /*is_def_allowed=*/false);
}

}  // namespace Carbon::Parse
