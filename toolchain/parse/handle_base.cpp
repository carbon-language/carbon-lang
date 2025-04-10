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
    // TODO: If the next token isn't a colon or `class`, try to recover
    // based on whether we're in a class, whether we have an `extend`
    // modifier, and the following tokens.
    CARBON_DIAGNOSTIC(ExpectedAfterBase, Error,
                      "`class` or `:` expected after `base`");
    context.emitter().Emit(*context.position(), ExpectedAfterBase);
    context.RecoverFromDeclError(state, NodeKind::BaseDecl,
                                 /*skip_past_likely_end=*/true);
    return;
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
