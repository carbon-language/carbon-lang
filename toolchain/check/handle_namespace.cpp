// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/decl_introducer_state.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/check/name_component.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/type.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::NamespaceStartId /*node_id*/)
    -> bool {
  // Optional modifiers and the name follow.
  context.decl_introducer_state_stack().Push<Lex::TokenKind::Namespace>();
  context.decl_name_stack().PushScopeAndStartName();
  return true;
}

static auto IsNamespaceScope(Context& context, SemIR::NameScopeId name_scope_id)
    -> bool {
  auto [_, inst] = context.name_scopes().GetInstIfValid(name_scope_id);
  return inst && inst->Is<SemIR::Namespace>();
}

auto HandleParseNode(Context& context, Parse::NamespaceId node_id) -> bool {
  auto name_context = context.decl_name_stack().FinishName(
      PopNameComponentWithoutParams(context, Lex::TokenKind::Namespace));

  auto introducer =
      context.decl_introducer_state_stack().Pop<Lex::TokenKind::Namespace>();
  LimitModifiersOnDecl(context, introducer, KeywordModifierSet::None);

  auto namespace_inst = SemIR::Namespace{
      GetSingletonType(context, SemIR::NamespaceType::SingletonInstId),
      SemIR::NameScopeId::None, SemIR::InstId::None};
  auto namespace_id = AddPlaceholderInst(context, node_id, namespace_inst);

  SemIR::ScopeLookupResult lookup_result =
      context.decl_name_stack().LookupOrAddName(name_context, namespace_id,
                                                SemIR::AccessKind::Public);
  if (lookup_result.is_poisoned()) {
    DiagnosePoisonedName(context, name_context.name_id_for_new_inst(),
                         lookup_result.poisoning_loc_id(), name_context.loc_id);
  } else if (lookup_result.is_found()) {
    SemIR::InstId existing_inst_id = lookup_result.target_inst_id();
    if (auto existing =
            context.insts().TryGetAs<SemIR::Namespace>(existing_inst_id)) {
      // If there's a name conflict with a namespace, "merge" by using the
      // previous declaration. Otherwise, diagnose the issue.

      // Point at the other namespace.
      namespace_inst.name_scope_id = existing->name_scope_id;

      if (context.name_scopes()
              .Get(existing->name_scope_id)
              .is_closed_import()) {
        // The existing name is a package name, so this is a name conflict.
        DiagnoseDuplicateName(context, name_context.name_id,
                              name_context.loc_id, existing_inst_id);

        // Treat this as a local namespace name from now on to avoid further
        // diagnostics.
        context.name_scopes()
            .Get(existing->name_scope_id)
            .set_is_closed_import(false);
      } else if (existing->import_id.has_value() &&
                 !context.insts().GetLocId(existing_inst_id).has_value()) {
        // When the name conflict is an imported namespace, fill the location ID
        // so that future diagnostics point at this declaration.
        SetNamespaceNodeId(context, existing_inst_id, node_id);
      }
    } else {
      DiagnoseDuplicateName(context, name_context.name_id, name_context.loc_id,
                            existing_inst_id);
    }
  }

  // If we weren't able to merge namespaces, add a new name scope. Note this
  // occurs even for duplicates where we discard the namespace, because we want
  // to produce a valid constant.
  if (!namespace_inst.name_scope_id.has_value()) {
    namespace_inst.name_scope_id = context.name_scopes().Add(
        namespace_id, name_context.name_id_for_new_inst(),
        name_context.parent_scope_id);
    if (!IsNamespaceScope(context, name_context.parent_scope_id)) {
      CARBON_DIAGNOSTIC(NamespaceDeclNotAtTopLevel, Error,
                        "`namespace` declaration not at top level");
      context.emitter().Emit(node_id, NamespaceDeclNotAtTopLevel);
    }
  }

  ReplaceInstBeforeConstantUse(context, namespace_id, namespace_inst);

  context.decl_name_stack().PopScope();
  return true;
}

}  // namespace Carbon::Check
