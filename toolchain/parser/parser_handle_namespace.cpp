// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_context.h"

namespace Carbon {

auto ParserHandleNamespace(ParserContext& context) -> void {
  auto state = context.PopState();

  context.AddLeafNode(ParseNodeKind::NamespaceStart, context.Consume());

  state.state = ParserState::NamespaceFinish;
  context.PushState(state);

  context.PushState(ParserState::DeclarationNameAndParamsAsNone, state.token);
}

auto ParserHandleNamespaceFinish(ParserContext& context) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    context.RecoverFromDeclarationError(state, ParseNodeKind::Namespace,
                                        /*skip_past_likely_end=*/true);
    return;
  }

  if (auto semi = context.ConsumeIf(TokenKind::Semi)) {
    context.AddNode(ParseNodeKind::Namespace, *semi, state.subtree_start,
                    state.has_error);
  } else {
    context.EmitExpectedDeclarationSemi(TokenKind::Namespace);
    context.RecoverFromDeclarationError(state, ParseNodeKind::Namespace,
                                        /*skip_past_likely_end=*/true);
  }
}

}  // namespace Carbon
