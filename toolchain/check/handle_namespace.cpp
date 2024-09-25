// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/decl_introducer_state.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/check/name_component.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/name_scope.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::NamespaceStartId /*node_id*/)
    -> bool {
  // Optional modifiers and the name follow.
  context.decl_introducer_state_stack().Push<Lex::TokenKind::Namespace>();
  context.decl_name_stack().PushScopeAndStartName();

  // Push a pattern block to handle parameters of the namespace declaration.
  // TODO: Disallow these in parse, instead of check, so we don't have to do
  // this.
  context.pattern_block_stack().Push();
  return true;
}

auto HandleParseNode(Context& context, Parse::NamespaceId node_id) -> bool {
  auto name_context = context.decl_name_stack().FinishName(
      PopNameComponentWithoutParams(context, Lex::TokenKind::Namespace));

  auto introducer =
      context.decl_introducer_state_stack().Pop<Lex::TokenKind::Namespace>();
  LimitModifiersOnDecl(context, introducer, KeywordModifierSet::None);

  auto namespace_inst = SemIR::Namespace{
      context.GetBuiltinType(SemIR::BuiltinInstKind::NamespaceType),
      SemIR::NameScopeId::Invalid, SemIR::InstId::Invalid};
  auto namespace_id =
      context.AddPlaceholderInst(SemIR::LocIdAndInst(node_id, namespace_inst));

  auto existing_inst_id = context.decl_name_stack().LookupOrAddName(
      name_context, namespace_id, SemIR::AccessKind::Public);
  if (existing_inst_id.is_valid()) {
    // If there's a name conflict with a namespace, "merge" by using the
    // previous declaration. Otherwise, diagnose the issue.
    if (auto existing =
            context.insts().TryGetAs<SemIR::Namespace>(existing_inst_id)) {
      // Point at the other namespace.
      namespace_inst.name_scope_id = existing->name_scope_id;

      if (context.name_scopes().Get(existing->name_scope_id).is_closed_import) {
        // The existing name is a package name, so this is a name conflict.
        context.DiagnoseDuplicateName(namespace_id, existing_inst_id);

        // Treat this as a local namespace name from now on to avoid further
        // diagnostics.
        context.name_scopes().Get(existing->name_scope_id).is_closed_import =
            false;
      } else if (existing->import_id.is_valid() &&
                 !context.insts().GetLocId(existing_inst_id).is_valid()) {
        // When the name conflict is an imported namespace, fill the location ID
        // so that future diagnostics point at this declaration.
        context.SetNamespaceNodeId(existing_inst_id, node_id);
      }
    } else {
      context.DiagnoseDuplicateName(namespace_id, existing_inst_id);
    }
  }

  // If we weren't able to merge namespaces, add a new name scope. Note this
  // occurs even for duplicates where we discard the namespace, because we want
  // to produce a valid constant.
  if (!namespace_inst.name_scope_id.is_valid()) {
    namespace_inst.name_scope_id = context.name_scopes().Add(
        namespace_id, name_context.name_id_for_new_inst(),
        name_context.parent_scope_id);
  }

  context.ReplaceInstBeforeConstantUse(namespace_id, namespace_inst);

  context.decl_name_stack().PopScope();
  return true;
}

}  // namespace Carbon::Check
