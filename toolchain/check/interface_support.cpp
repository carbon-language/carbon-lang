// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/interface_support.h"

#include "toolchain/check/inst.h"
#include "toolchain/check/merge.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/sem_ir/entity_with_params_base.h"

namespace Carbon::Check {

auto GetExistingDeclOrDiagnoseMismatch(
    Context& context, Parse::NodeId node_id, const NameComponent& name,
    const DeclNameStack::NameContext& name_context,
    const SemIR::EntityWithParamsBase& entity, bool is_definition,
    llvm::function_ref<auto(SemIR::Inst)->const SemIR::EntityWithParamsBase*>
        try_get_entity,
    SemIR::ScopeLookupResult lookup_result) -> std::optional<SemIR::Inst> {
  if (lookup_result.is_poisoned()) {
    // This is a declaration of a poisoned name.
    DiagnosePoisonedName(context, name_context.name_id_for_new_inst(),
                         lookup_result.poisoning_loc_id(), name_context.loc_id);
    return {};
  }

  if (!lookup_result.is_found()) {
    return {};
  }

  SemIR::InstId existing_id = lookup_result.target_inst_id();
  SemIR::Inst existing_decl_inst = context.insts().Get(existing_id);
  const auto* existing_decl_entity = try_get_entity(existing_decl_inst);
  if (!existing_decl_entity) {
    // This is a redeclaration of something other than a interface.
    DiagnoseDuplicateName(context, name_context.name_id, name_context.loc_id,
                          SemIR::LocId(existing_id));
    return {};
  }

  if (CheckRedeclParamsMatch(
          context,
          DeclParams(node_id, name.first_param_node_id, name.last_param_node_id,
                     name.implicit_param_patterns_id, name.param_patterns_id),
          DeclParams(*existing_decl_entity))) {
    // TODO: This should be refactored a little, particularly for
    // prev_import_ir_id. See similar logic for classes and functions, which
    // might also be refactored to merge.
    DiagnoseIfInvalidRedecl(
        context, Lex::TokenKind::Interface, existing_decl_entity->name_id,
        RedeclInfo(entity, node_id, is_definition),
        RedeclInfo(*existing_decl_entity,
                   SemIR::LocId(existing_decl_entity->latest_decl_id()),
                   existing_decl_entity->has_definition_started()),
        /*prev_import_ir_id=*/SemIR::ImportIRId::None);

    // Can't merge definitions due to the generic requirements.
    if (!is_definition || !existing_decl_entity->has_definition_started()) {
      // This is a redeclaration of an existing entity of the same type.
      return existing_decl_inst;
    }
  }
  return {};
}

auto GetSelfParameter(Context& context, SemIR::TypeId type_id,
                      SemIR::NameScopeId scope_id, bool is_template)
    -> SemIR::InstId {
  auto entity_name_id = context.entity_names().AddSymbolicBindingName(
      SemIR::NameId::SelfType, scope_id,
      context.scope_stack().AddCompileTimeBinding(), is_template);
  // Because there is no equivalent non-symbolic value, we use `None` as
  // the `value_id` on the `BindSymbolicName`.
  auto self_param_inst_id =
      AddInst(context, SemIR::LocIdAndInst::NoLoc<SemIR::BindSymbolicName>(
                           {.type_id = type_id,
                            .entity_name_id = entity_name_id,
                            .value_id = SemIR::InstId::None}));
  context.scope_stack().PushCompileTimeBinding(self_param_inst_id);
  context.name_scopes().AddRequiredName(scope_id, SemIR::NameId::SelfType,
                                        self_param_inst_id);
  return self_param_inst_id;
}

}  // namespace Carbon::Check
