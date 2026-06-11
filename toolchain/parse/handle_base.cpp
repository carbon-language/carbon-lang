// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

// Handles a `base` declaration after the introducer.
auto HandleBaseAfterIntroducer(Context& context) -> void {
  auto state = context.PopState();

  if (!context.ConsumeAndAddLeafNodeIf(Lex::TokenKind::Colon,
                                       NodeKind::BaseColon)) {
    CARBON_DIAGNOSTIC(ExpectedAfterBase, Error,
                      "`class` or `:` expected after `base`");
    context.emitter().Emit(*context.position(), ExpectedAfterBase);
    auto base_token = *(context.position() - 1);
    auto previous_token = Lex::TokenIndex(base_token.index - 1);
    if (context.tokens().GetKind(previous_token) != Lex::TokenKind::Extend) {
      context.RecoverFromDeclError(state, NodeKind::BaseDecl,
                                   /*skip_past_likely_end=*/true);
      return;
    }

    // Preserve the `extend base` tree shape using an errored placeholder.
    context.AddLeafNode(NodeKind::BaseColon, *context.position(),
                        /*has_error=*/true);
    state.has_error = true;
  }

  state.kind = StateKind::BaseDecl;
  context.PushState(state);
  context.PushState(StateKind::Expr);
}

// Handles processing of a complete `base: B` declaration.
auto HandleBaseDecl(Context& context) -> void {
  auto state = context.PopState();

  context.AddNodeExpectingDeclSemi(state, NodeKind::BaseDecl,
                                   Lex::TokenKind::Base,
                                   /*is_def_allowed=*/false);
}

}  // namespace Carbon::Parse
