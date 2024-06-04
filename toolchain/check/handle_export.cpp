// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/check/name_component.h"
#include "toolchain/parse/typed_nodes.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleExportIntroducer(Context& context,
                            Parse::ExportIntroducerId /*node_id*/) -> bool {
  context.decl_introducer_state_stack().Push(DeclIntroducerState::Export);
  // TODO: Probably need to update DeclNameStack to restrict to only namespaces.
  context.decl_name_stack().PushScopeAndStartName();
  return true;
}

auto HandleExportDecl(Context& context, Parse::ExportDeclId node_id) -> bool {
  auto name_context = context.decl_name_stack().FinishName(
      PopNameComponentWithoutParams(context, Lex::TokenKind::Export));
  context.decl_name_stack().PopScope();

  auto introducer =
      context.decl_introducer_state_stack().Pop(DeclIntroducerState::Export);
  LimitModifiersOnDecl(context, introducer, KeywordModifierSet::None,
                       Lex::TokenKind::Export);

  if (name_context.state == DeclNameStack::NameContext::State::Error) {
    // Should already be diagnosed.
    return true;
  }

  auto inst_id = name_context.prev_inst_id();
  if (!inst_id.is_valid()) {
    context.DiagnoseNameNotFound(node_id, name_context.name_id_for_new_inst());
    return true;
  }

  auto inst = context.insts().Get(inst_id);

  if (inst.Is<SemIR::ExportDecl>()) {
    CARBON_DIAGNOSTIC(ExportRedundant, Warning,
                      "`export` matches previous `export`.");
    CARBON_DIAGNOSTIC(ExportPrevious, Note, "Previous `export` here.");
    context.emitter()
        .Build(node_id, ExportRedundant)
        // Use the location of the export itself, not the exported instruction.
        .Note(context.insts().GetLocId(inst_id), ExportPrevious)
        .Emit();
    return true;
  }

  auto import_ref = context.insts().TryGetAs<SemIR::ImportRefLoaded>(inst_id);
  if (!import_ref) {
    CARBON_DIAGNOSTIC(ExportNotImportedEntity, Error,
                      "Only imported entities are valid for `export`.");
    CARBON_DIAGNOSTIC(ExportNotImportedEntitySource, Note,
                      "Name is declared here.");
    context.emitter()
        .Build(node_id, ExportNotImportedEntity)
        .Note(inst_id, ExportNotImportedEntitySource)
        .Emit();
    return true;
  }

  auto export_id = context.AddInst<SemIR::ExportDecl>(
      node_id, {.type_id = import_ref->type_id,
                .bind_name_id = import_ref->bind_name_id,
                .value_id = inst_id});
  context.AddExport(export_id);

  // Replace the ImportRef in name lookup, both for the above duplicate
  // diagnostic and so that cross-package imports can find it easily.
  auto bind_name = context.bind_names().Get(import_ref->bind_name_id);
  auto& names = context.name_scopes().Get(bind_name.parent_scope_id).names;
  auto it = names.find(bind_name.name_id);
  CARBON_CHECK(it->second == inst_id);
  it->second = export_id;

  return true;
}

}  // namespace Carbon::Check
