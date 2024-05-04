// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/class.h"

#include "toolchain/check/merge.h"

namespace Carbon::Check {

auto MergeClassRedecl(Context& context, SemIRLoc new_loc,
                      SemIR::Class& new_class, bool new_is_import,
                      bool new_is_definition, bool new_is_extern,
                      SemIR::ClassId prev_class_id, bool prev_is_extern,
                      SemIR::ImportIRInstId prev_import_ir_inst_id) -> bool {
  auto& prev_class = context.classes().Get(prev_class_id);
  SemIRLoc prev_loc =
      prev_class.is_defined() ? prev_class.definition_id : prev_class.decl_id;

  // Check the generic parameters match, if they were specified.
  if (!CheckRedeclParamsMatch(context, DeclParams(new_class),
                              DeclParams(prev_class), {})) {
    return false;
  }

  CheckIsAllowedRedecl(context, Lex::TokenKind::Class, prev_class.name_id,
                       {.loc = new_loc,
                        .is_definition = new_is_definition,
                        .is_extern = new_is_extern},
                       {.loc = prev_loc,
                        .is_definition = prev_class.is_defined(),
                        .is_extern = prev_is_extern},
                       prev_import_ir_inst_id);

  // The introducer kind must match the previous declaration.
  // TODO: The rule here is not yet decided. See #3384.
  if (prev_class.inheritance_kind != new_class.inheritance_kind) {
    CARBON_DIAGNOSTIC(ClassRedeclarationDifferentIntroducer, Error,
                      "Class redeclared with different inheritance kind.");
    CARBON_DIAGNOSTIC(ClassRedeclarationDifferentIntroducerPrevious, Note,
                      "Previously declared here.");
    context.emitter()
        .Build(new_loc, ClassRedeclarationDifferentIntroducer)
        .Note(prev_loc, ClassRedeclarationDifferentIntroducerPrevious)
        .Emit();
  }

  if (new_is_definition) {
    prev_class.implicit_param_refs_id = new_class.implicit_param_refs_id;
    prev_class.param_refs_id = new_class.param_refs_id;
    prev_class.definition_id = new_class.definition_id;
    prev_class.scope_id = new_class.scope_id;
    prev_class.body_block_id = new_class.body_block_id;
    prev_class.adapt_id = new_class.adapt_id;
    prev_class.base_id = new_class.base_id;
    prev_class.object_repr_id = new_class.object_repr_id;
  }

  if ((prev_import_ir_inst_id.is_valid() && !new_is_import) ||
      (prev_is_extern && !new_is_extern)) {
    prev_class.decl_id = new_class.decl_id;
    ReplacePrevInstForMerge(
        context, prev_class.enclosing_scope_id, prev_class.name_id,
        new_is_import ? new_loc.inst_id : new_class.decl_id);
  }
  return true;
}

}  // namespace Carbon::Check
