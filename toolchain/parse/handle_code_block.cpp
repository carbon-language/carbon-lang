// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

auto HandleCodeBlock(Context& context) -> void {
  context.PopAndDiscardState();

  context.PushState(StateKind::CodeBlockFinish);
  if (context.ConsumeAndAddLeafNodeIf(Lex::TokenKind::OpenCurlyBrace,
                                      NodeKind::CodeBlockStart)) {
    context.PushState(StateKind::StatementScopeLoop);
  } else {
    context.AddLeafNode(NodeKind::CodeBlockStart, *context.position(),
                        /*has_error=*/true);

    // Recover by parsing a single statement.
    CARBON_DIAGNOSTIC(ExpectedCodeBlock, Error, "expected braced code block");
    context.emitter().Emit(*context.position(), ExpectedCodeBlock);

    context.PushState(StateKind::Statement);
  }
}

auto HandleCodeBlockFinish(Context& context) -> void {
  auto state = context.PopState();

  // If the block started with an open curly, this is a close curly.
  if (context.tokens().GetKind(state.token) == Lex::TokenKind::OpenCurlyBrace) {
    context.AddNode(NodeKind::CodeBlock, context.Consume(), state.has_error);
  } else {
    context.AddNode(NodeKind::CodeBlock, state.token, /*has_error=*/true);
  }
}

}  // namespace Carbon::Parse
