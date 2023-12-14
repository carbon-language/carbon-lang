// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/decl_state.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

auto HandleNamespaceStart(Context& context, Parse::NodeId parse_node) -> bool {
  // Optional modifiers and the name follow.
  context.decl_state_stack().Push(DeclState::Namespace, parse_node);
  context.decl_name_stack().PushScopeAndStartName();
  return true;
}

auto HandleNamespace(Context& context, Parse::NodeId parse_node) -> bool {
  auto name_context = context.decl_name_stack().FinishName();
  LimitModifiersOnDecl(context, KeywordModifierSet::None,
                       Lex::TokenKind::Namespace);
  auto namespace_id = context.AddInst(SemIR::Namespace{
      parse_node, context.GetBuiltinType(SemIR::BuiltinKind::NamespaceType),
      context.name_scopes().Add()});
  context.decl_name_stack().AddNameToLookup(name_context, namespace_id);

  context.decl_name_stack().PopScope();
  context.decl_state_stack().Pop(DeclState::Namespace);
  return true;
}

}  // namespace Carbon::Check
