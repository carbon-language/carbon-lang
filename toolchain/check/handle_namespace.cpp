// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/decl_state.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

auto HandleNamespaceStart(Context& context,
                          Parse::NamespaceStartId /*parse_node*/) -> bool {
  // Optional modifiers and the name follow.
  context.decl_state_stack().Push(DeclState::Namespace);
  context.decl_name_stack().PushScopeAndStartName();
  return true;
}

auto HandleNamespace(Context& context, Parse::NamespaceId parse_node) -> bool {
  auto name_context = context.decl_name_stack().FinishName();
  LimitModifiersOnDecl(context, KeywordModifierSet::None,
                       Lex::TokenKind::Namespace);
  auto namespace_inst = SemIR::Namespace{
      context.GetBuiltinType(SemIR::BuiltinKind::NamespaceType),
      name_context.name_id_for_new_inst(), SemIR::NameScopeId::Invalid};
  auto namespace_id = context.AddPlaceholderInst({parse_node, namespace_inst});
  namespace_inst.name_scope_id = context.name_scopes().Add(
      namespace_id, name_context.enclosing_scope_id_for_new_inst());
  context.ReplaceInstBeforeConstantUse(namespace_id,
                                       {parse_node, namespace_inst});
  context.decl_name_stack().AddNameToLookup(name_context, namespace_id);

  context.decl_name_stack().PopScope();
  context.decl_state_stack().Pop(DeclState::Namespace);
  return true;
}

}  // namespace Carbon::Check
