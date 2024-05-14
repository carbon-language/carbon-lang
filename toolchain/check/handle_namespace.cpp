// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/decl_state.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

auto HandleNamespaceStart(Context& context, Parse::NamespaceStartId /*node_id*/)
    -> bool {
  // Optional modifiers and the name follow.
  context.decl_state_stack().Push(DeclState::Namespace);
  context.decl_name_stack().PushScopeAndStartName();
  return true;
}

auto HandleNamespace(Context& context, Parse::NamespaceId node_id) -> bool {
  auto name_context = context.decl_name_stack().FinishName();
  LimitModifiersOnDecl(context, KeywordModifierSet::None,
                       Lex::TokenKind::Namespace);
  auto namespace_inst = SemIR::Namespace{
      context.GetBuiltinType(SemIR::BuiltinKind::NamespaceType),
      SemIR::NameScopeId::Invalid, SemIR::InstId::Invalid};
  auto namespace_id = context.AddPlaceholderInst({node_id, namespace_inst});
  namespace_inst.name_scope_id = context.name_scopes().Add(
      namespace_id, name_context.name_id_for_new_inst(),
      name_context.enclosing_scope_id_for_new_inst());
  context.ReplaceInstBeforeConstantUse(namespace_id, namespace_inst);

  auto existing_inst_id =
      context.decl_name_stack().LookupOrAddName(name_context, namespace_id);
  if (existing_inst_id.is_valid()) {
    // If there's a name conflict with a namespace, "merge" by using the
    // previous declaration. Otherwise, diagnose the issue.
    if (auto existing =
            context.insts().TryGetAs<SemIR::Namespace>(existing_inst_id)) {
      // When the name conflict is an imported namespace, fill the location ID
      // so that future diagnostics point at this declaration.
      if (existing->import_id.is_valid() &&
          !context.insts().GetLocId(existing_inst_id).is_valid()) {
        context.SetNamespaceNodeId(existing_inst_id, node_id);
      }
    } else {
      context.DiagnoseDuplicateName(namespace_id, existing_inst_id);
    }
  }

  context.decl_name_stack().PopScope();
  context.decl_state_stack().Pop(DeclState::Namespace);
  return true;
}

}  // namespace Carbon::Check
