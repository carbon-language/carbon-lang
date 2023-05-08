// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_context.h"

namespace Carbon {

auto ParserHandleCodeBlock(ParserContext& context) -> void {
  context.PopAndDiscardState();

  context.PushState(ParserState::CodeBlockFinish);
  if (context.ConsumeAndAddLeafNodeIf(TokenKind::OpenCurlyBrace,
                                      ParseNodeKind::CodeBlockStart)) {
    context.PushState(ParserState::StatementScopeLoop);
  } else {
    context.AddLeafNode(ParseNodeKind::CodeBlockStart, *context.position(),
                        /*has_error=*/true);

    // Recover by parsing a single statement.
    CARBON_DIAGNOSTIC(ExpectedCodeBlock, Error, "Expected braced code block.");
    context.emitter().Emit(*context.position(), ExpectedCodeBlock);

    context.PushState(ParserState::Statement);
  }
}

auto ParserHandleCodeBlockFinish(ParserContext& context) -> void {
  auto state = context.PopState();

  // If the block started with an open curly, this is a close curly.
  if (context.tokens().GetKind(state.token) == TokenKind::OpenCurlyBrace) {
    context.AddNode(ParseNodeKind::CodeBlock, context.Consume(),
                    state.subtree_start, state.has_error);
  } else {
    context.AddNode(ParseNodeKind::CodeBlock, state.token, state.subtree_start,
                    /*has_error=*/true);
  }
}

}  // namespace Carbon
