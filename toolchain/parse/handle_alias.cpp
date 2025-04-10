// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/token_kind.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/parse/state.h"

namespace Carbon::Parse {

auto HandleAlias(Context& context) -> void {
  auto state = context.PopState();

  context.PushState(state, StateKind::AliasAfterName);
  context.PushState(StateKind::DeclNameAndParams, state.token);
}

auto HandleAliasAfterName(Context& context) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    context.RecoverFromDeclError(state, NodeKind::Alias,
                                 /*skip_past_likely_end=*/true);
    return;
  }

  if (auto equal = context.ConsumeIf(Lex::TokenKind::Equal)) {
    context.AddLeafNode(NodeKind::AliasInitializer, *equal);
    context.PushState(state, StateKind::AliasFinish);
    context.PushState(StateKind::Expr);
  } else {
    CARBON_DIAGNOSTIC(ExpectedAliasInitializer, Error,
                      "`alias` requires a `=` for the source");
    context.emitter().Emit(*context.position(), ExpectedAliasInitializer);
    context.RecoverFromDeclError(state, NodeKind::Alias,
                                 /*skip_past_likely_end=*/true);
  }
}

auto HandleAliasFinish(Context& context) -> void {
  auto state = context.PopState();

  context.AddNodeExpectingDeclSemi(state, NodeKind::Alias,
                                   Lex::TokenKind::Alias,
                                   /*is_def_allowed=*/false);
}

}  // namespace Carbon::Parse
