// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_context.h"

namespace Carbon {

// Handles DesignatorAs.
auto ParserHandleDesignator(ParserContext& context, bool as_struct) -> void {
  auto state = context.PopState();

  // `.` identifier
  auto dot = context.ConsumeChecked(TokenKind::Period);

  if (!context.ConsumeAndAddLeafNodeIf(TokenKind::Identifier,
                                       ParseNodeKind::DesignatedName)) {
    CARBON_DIAGNOSTIC(ExpectedIdentifierAfterDot, Error,
                      "Expected identifier after `.`.");
    context.emitter().Emit(*context.position(), ExpectedIdentifierAfterDot);
    // If we see a keyword, assume it was intended to be the designated name.
    // TODO: Should keywords be valid in designators?
    if (context.PositionKind().is_keyword()) {
      context.AddLeafNode(ParseNodeKind::DesignatedName, context.Consume(),
                          /*has_error=*/true);
    } else {
      context.AddLeafNode(ParseNodeKind::DesignatedName, *context.position(),
                          /*has_error=*/true);
      // Indicate the error to the parent state so that it can avoid producing
      // more errors.
      context.ReturnErrorOnState();
    }
  }

  context.AddNode(as_struct ? ParseNodeKind::StructFieldDesignator
                            : ParseNodeKind::DesignatorExpression,
                  dot, state.subtree_start, state.has_error);
}

auto ParserHandleDesignatorAsExpression(ParserContext& context) -> void {
  ParserHandleDesignator(context, /*as_struct=*/false);
}

auto ParserHandleDesignatorAsStruct(ParserContext& context) -> void {
  ParserHandleDesignator(context, /*as_struct=*/true);
}

}  // namespace Carbon
