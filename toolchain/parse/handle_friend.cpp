// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/format_providers.h"
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
    // A word that can't be a name, followed by the `;`, was most likely meant
    // as the name, and is what the message is about. Otherwise the name is
    // missing, and goes after `friend`.
    bool recover_as_raw =
        context.PositionKind().is_word() &&
        context.PositionKind(Lookahead::NextToken) == Lex::TokenKind::Semi;
    CARBON_DIAGNOSTIC(
        ExpectedIdentifierForFriend, Error,
        "expected identifier for friend declaration"
        "{0:; prefix reserved word with `r#` to form a valid identifier|}",
        Diagnostics::BoolAsSelect);
    context.emitter().Emit(recover_as_raw ? DiagnosticLoc(*context.position())
                                          : context.PointAfterPreviousToken(),
                           ExpectedIdentifierForFriend, recover_as_raw);
    if (!recover_as_raw) {
      state.has_error = true;
      context.RecoverFromDeclError(state, NodeKind::FriendDecl,
                                   /*skip_past_likely_end=*/true);
      return;
    }
    context.AddLeafNode(NodeKind::IdentifierNameExpr,
                        context.tokens().AddPostLexingRecoveryTokenAsIdentifier(
                            context.Consume()));
  }
  context.AddNodeExpectingDeclSemi(state, NodeKind::FriendDecl,
                                   Lex::TokenKind::Friend,
                                   /*is_def_allowed=*/false);
}

}  // namespace Carbon::Parse
