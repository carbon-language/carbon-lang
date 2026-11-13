// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/parse/precedence.h"
#include "toolchain/parse/state.h"
#include "toolchain/parse/typed_nodes.h"

namespace Carbon::Parse {
auto HandleWhereAssignment(Context& context) -> void {
  auto state = context.PopState();

  if (!context.ConsumeAndAddLeafNodeIf(Lex::TokenKind::Identifier,
                                       NodeKind::IdentifierNameExpr)) {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(
          WhereAssignmentMissingIdentifier, Error,
          "expected identifier in `where` assignment declaration");
      context.emitter().Emit(context.PointAfterPreviousToken(),
                             WhereAssignmentMissingIdentifier);
    }
    context.RecoverFromDeclError(state, NodeKind::WhereAssignmentDecl,
                                 /*skip_past_likely_end=*/true);
    return;
  }

  auto eq_token = context.ConsumeIf(Lex::TokenKind::Equal);
  if (!eq_token) {
    if (!state.has_error) {
      CARBON_DIAGNOSTIC(WhereAssignmentMissingEquals, Error,
                        "missing `=` in `where` assignment");
      context.emitter().Emit(context.PointAfterPreviousToken(),
                             WhereAssignmentMissingEquals);
    }
    context.RecoverFromDeclError(state, NodeKind::WhereAssignmentDecl,
                                 /*skip_past_likely_end=*/true);
    return;
  }

  state.token = *eq_token;
  context.PushState(state, StateKind::WhereAssignmentDecl);
  context.PushStateForExpr(PrecedenceGroup::ForRequirements());
}

auto HandleWhereAssignmentDecl(Context& context) -> void {
  auto state = context.PopState();
  context.AddNode(NodeKind::WhereAssignmentOperator, state.token,
                  state.has_error);
  context.AddNodeExpectingDeclSemi(state, NodeKind::WhereAssignmentDecl,
                                   Lex::TokenKind::Where,
                                   /*is_def_allowed=*/false);
}
}  // namespace Carbon::Parse
