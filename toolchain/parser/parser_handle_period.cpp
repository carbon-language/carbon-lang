// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_context.h"

namespace Carbon {

// Handles PeriodAs variants and ArrowExpression.
// TODO: This currently only supports identifiers on the rhs, but will in the
// future need to handle things like `object.(Interface.member)` for qualifiers.
auto ParserHandlePeriodOrArrow(ParserContext& context, ParseNodeKind node_kind,
                               bool is_arrow) -> void {
  auto state = context.PopState();

  // `.` identifier
  auto dot = context.ConsumeChecked(is_arrow ? TokenKind::MinusGreater
                                             : TokenKind::Period);

  if (!context.ConsumeAndAddLeafNodeIf(TokenKind::Identifier,
                                       ParseNodeKind::Name)) {
    CARBON_DIAGNOSTIC(ExpectedIdentifierAfterDotOrArrow, Error,
                      "Expected identifier after `{0}`.", llvm::StringRef);
    context.emitter().Emit(*context.position(),
                           ExpectedIdentifierAfterDotOrArrow,
                           is_arrow ? "->" : ".");
    // If we see a keyword, assume it was intended to be a name.
    // TODO: Should keywords be valid here?
    if (context.PositionKind().is_keyword()) {
      context.AddLeafNode(ParseNodeKind::Name, context.Consume(),
                          /*has_error=*/true);
    } else {
      context.AddLeafNode(ParseNodeKind::Name, *context.position(),
                          /*has_error=*/true);
      // Indicate the error to the parent state so that it can avoid producing
      // more errors.
      context.ReturnErrorOnState();
    }
  }

  context.AddNode(node_kind, dot, state.subtree_start, state.has_error);
}

auto ParserHandlePeriodAsDeclaration(ParserContext& context) -> void {
  ParserHandlePeriodOrArrow(context, ParseNodeKind::QualifiedDeclaration,
                            false);
}

auto ParserHandlePeriodAsExpression(ParserContext& context) -> void {
  ParserHandlePeriodOrArrow(context, ParseNodeKind::MemberAccessExpression,
                            false);
}

auto ParserHandlePeriodAsStruct(ParserContext& context) -> void {
  ParserHandlePeriodOrArrow(context, ParseNodeKind::StructFieldDesignator,
                            false);
}

auto ParserHandleArrowExpression(ParserContext& context) -> void {
  ParserHandlePeriodOrArrow(context,
                            ParseNodeKind::PointerMemberAccessExpression, true);
}

}  // namespace Carbon
