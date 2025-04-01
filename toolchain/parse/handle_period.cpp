// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

// Handles PeriodAs variants and ArrowExpr.
// TODO: This currently only supports identifiers on the rhs, but will in the
// future need to handle things like `object.(Interface.member)` for qualifiers.
static auto HandlePeriodOrArrow(Context& context, NodeKind node_kind,
                                State paren_state, bool is_arrow) -> void {
  auto state = context.PopState();

  // We're handling `.something` or `->something`.
  auto dot = context.ConsumeChecked(is_arrow ? Lex::TokenKind::MinusGreater
                                             : Lex::TokenKind::Period);

  if (context.ConsumeAndAddLeafNodeIf(
          Lex::TokenKind::Identifier,
          NodeKind::IdentifierNameNotBeforeParams)) {
    // OK, `.` identifier.
  } else if (context.ConsumeAndAddLeafNodeIf(Lex::TokenKind::Base,
                                             NodeKind::BaseName)) {
    // OK, `.base`.
  } else if (node_kind != NodeKind::StructFieldDesignator &&
             context.ConsumeAndAddLeafNodeIf(Lex::TokenKind::IntLiteral,
                                             NodeKind::IntLiteral)) {
    // OK, '.42'.
  } else if (paren_state != State::Invalid &&
             context.PositionIs(Lex::TokenKind::OpenParen)) {
    state.state = paren_state;
    context.PushState(state);
    context.PushState(State::OnlyParenExpr);
    return;
  } else {
    CARBON_DIAGNOSTIC(ExpectedIdentifierAfterPeriodOrArrow, Error,
                      "expected identifier after `{0:->|.}`",
                      Diagnostics::BoolAsSelect);
    context.emitter().Emit(*context.position(),
                           ExpectedIdentifierAfterPeriodOrArrow, is_arrow);
    // If we see a keyword, assume it was intended to be a name.
    // TODO: Should keywords be valid here?
    if (context.PositionKind().is_keyword()) {
      context.AddLeafNode(NodeKind::IdentifierNameNotBeforeParams,
                          context.Consume(), /*has_error=*/true);
    } else {
      context.AddLeafNode(NodeKind::IdentifierNameNotBeforeParams,
                          *context.position(), /*has_error=*/true);
      // Indicate the error to the parent state so that it can avoid producing
      // more errors.
      context.ReturnErrorOnState();
    }
  }

  context.AddNode(node_kind, dot, state.has_error);
}

auto HandlePeriodAsExpr(Context& context) -> void {
  HandlePeriodOrArrow(context, NodeKind::MemberAccessExpr,
                      State::CompoundMemberAccess,
                      /*is_arrow=*/false);
}

auto HandlePeriodAsStruct(Context& context) -> void {
  HandlePeriodOrArrow(context, NodeKind::StructFieldDesignator, State::Invalid,
                      /*is_arrow=*/false);
}

auto HandleArrowExpr(Context& context) -> void {
  HandlePeriodOrArrow(context, NodeKind::PointerMemberAccessExpr,
                      State::CompoundPointerMemberAccess,
                      /*is_arrow=*/true);
}

auto HandleCompoundMemberAccess(Context& context) -> void {
  auto state = context.PopState();
  context.AddNode(NodeKind::MemberAccessExpr, state.token, state.has_error);
}

auto HandleCompoundPointerMemberAccess(Context& context) -> void {
  auto state = context.PopState();
  context.AddNode(NodeKind::PointerMemberAccessExpr, state.token,
                  state.has_error);
}

}  // namespace Carbon::Parse
