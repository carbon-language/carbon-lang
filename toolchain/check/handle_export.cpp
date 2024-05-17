// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/parse/typed_nodes.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleExportIntroducer(Context& context,
                            Parse::ExportIntroducerId /*node_id*/) -> bool {
  context.decl_state_stack().Push(DeclState::Export);
  // TODO: Probably need to update DeclNameStack to restrict to only namespaces.
  context.decl_name_stack().PushScopeAndStartName();
  return true;
}

auto HandleExportDirective(Context& context, Parse::ExportDirectiveId node_id)
    -> bool {
  auto name_context = context.decl_name_stack().FinishName();
  context.decl_name_stack().PopScope();

  LimitModifiersOnDecl(context, KeywordModifierSet::None,
                       Lex::TokenKind::Export);
  context.decl_state_stack().Pop(DeclState::Export);

  if (name_context.state == DeclNameStack::NameContext::State::Error) {
    // Should already be diagnosed.
    return true;
  }

  auto inst_id = name_context.prev_inst_id();
  if (!inst_id.is_valid()) {
    context.DiagnoseNameNotFound(node_id, name_context.name_id_for_new_inst());
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

  auto export_id = context.AddInst(
      {node_id, SemIR::BindExport{.type_id = import_ref->type_id,
                                  .bind_name_id = import_ref->bind_name_id,
                                  .value_id = inst_id}});
  context.AddExport(export_id);

  return true;
}

}  // namespace Carbon::Check
