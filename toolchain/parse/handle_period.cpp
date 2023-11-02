// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

// Handles PeriodAs variants and ArrowExpression.
// TODO: This currently only supports identifiers on the rhs, but will in the
// future need to handle things like `object.(Interface.member)` for qualifiers.
static auto HandlePeriodOrArrow(Context& context, NodeKind node_kind,
                                bool is_arrow) -> void {
  auto state = context.PopState();

  // `.` identifier
  auto dot = context.ConsumeChecked(is_arrow ? Lex::TokenKind::MinusGreater
                                             : Lex::TokenKind::Period);

  if (!context.ConsumeAndAddLeafNodeIf(Lex::TokenKind::Identifier,
                                       NodeKind::Name)) {
    CARBON_DIAGNOSTIC(ExpectedIdentifierAfterDotOrArrow, Error,
                      "Expected identifier after `{0}`.", llvm::StringRef);
    context.emitter().Emit(*context.position(),
                           ExpectedIdentifierAfterDotOrArrow,
                           is_arrow ? "->" : ".");
    // If we see a keyword, assume it was intended to be a name.
    // TODO: Should keywords be valid here?
    if (context.PositionKind().is_keyword()) {
      context.AddLeafNode(NodeKind::Name, context.Consume(),
                          /*has_error=*/true);
    } else {
      context.AddLeafNode(NodeKind::Name, *context.position(),
                          /*has_error=*/true);
      // Indicate the error to the parent state so that it can avoid producing
      // more errors.
      context.ReturnErrorOnState();
    }
  }

  context.AddNode(node_kind, dot, state.subtree_start, state.has_error);
}

auto HandlePeriodAsDeclaration(Context& context) -> void {
  HandlePeriodOrArrow(context, NodeKind::QualifiedDeclaration,
                      /*is_arrow=*/false);
}

auto HandlePeriodAsExpression(Context& context) -> void {
  HandlePeriodOrArrow(context, NodeKind::MemberAccessExpression,
                      /*is_arrow=*/false);
}

auto HandlePeriodAsStruct(Context& context) -> void {
  HandlePeriodOrArrow(context, NodeKind::StructFieldDesignator,
                      /*is_arrow=*/false);
}

auto HandleArrowExpression(Context& context) -> void {
  HandlePeriodOrArrow(context, NodeKind::PointerMemberAccessExpression,
                      /*is_arrow=*/true);
}

}  // namespace Carbon::Parse
