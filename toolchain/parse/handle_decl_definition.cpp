// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

// Handles processing after params, deciding whether it's a declaration or
// definition.
static auto HandleDeclOrDefinition(Context& context, NodeKind decl_kind,
                                   NodeKind definition_start_kind,
                                   StateKind definition_finish_state) -> void {
  auto state = context.PopState();

  if (state.has_error || !context.PositionIs(Lex::TokenKind::OpenCurlyBrace)) {
    context.AddNodeExpectingDeclSemi(state, decl_kind,
                                     context.tokens().GetKind(state.token),
                                     /*is_def_allowed=*/true);
    return;
  }

  context.PushState(state, definition_finish_state);
  context.PushState(StateKind::DeclScopeLoop);
  context.AddNode(definition_start_kind, context.Consume(), state.has_error);
}

auto HandleDeclOrDefinitionAsClass(Context& context) -> void {
  HandleDeclOrDefinition(context, NodeKind::ClassDecl,
                         NodeKind::ClassDefinitionStart,
                         StateKind::DeclDefinitionFinishAsClass);
}

auto HandleDeclOrDefinitionAsImpl(Context& context) -> void {
  HandleDeclOrDefinition(context, NodeKind::ImplDecl,
                         NodeKind::ImplDefinitionStart,
                         StateKind::DeclDefinitionFinishAsImpl);
}

auto HandleDeclOrDefinitionAsInterface(Context& context) -> void {
  HandleDeclOrDefinition(context, NodeKind::InterfaceDecl,
                         NodeKind::InterfaceDefinitionStart,
                         StateKind::DeclDefinitionFinishAsInterface);
}

auto HandleDeclOrDefinitionAsNamedConstraint(Context& context) -> void {
  HandleDeclOrDefinition(context, NodeKind::NamedConstraintDecl,
                         NodeKind::NamedConstraintDefinitionStart,
                         StateKind::DeclDefinitionFinishAsNamedConstraint);
}

// Handles parsing after the declaration scope of a type.
static auto HandleDeclDefinitionFinish(Context& context,
                                       NodeKind definition_kind) -> void {
  auto state = context.PopState();

  context.AddNode(definition_kind, context.Consume(), state.has_error);
}

auto HandleDeclDefinitionFinishAsClass(Context& context) -> void {
  HandleDeclDefinitionFinish(context, NodeKind::ClassDefinition);
}

auto HandleDeclDefinitionFinishAsImpl(Context& context) -> void {
  HandleDeclDefinitionFinish(context, NodeKind::ImplDefinition);
}

auto HandleDeclDefinitionFinishAsInterface(Context& context) -> void {
  HandleDeclDefinitionFinish(context, NodeKind::InterfaceDefinition);
}

auto HandleDeclDefinitionFinishAsNamedConstraint(Context& context) -> void {
  HandleDeclDefinitionFinish(context, NodeKind::NamedConstraintDefinition);
}

}  // namespace Carbon::Parse
