// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/class.h"

#include "toolchain/check/merge.h"

namespace Carbon::Check {

auto MergeClassRedecl(Context& context, SemIRLoc new_loc,
                      SemIR::Class& new_class, bool /*new_is_import*/,
                      bool new_is_definition, bool new_is_extern,
                      SemIR::ClassId prev_class_id, bool prev_is_extern,
                      SemIR::ImportIRInstId prev_import_ir_inst_id) -> bool {
  auto& prev_class = context.classes().Get(prev_class_id);
  SemIRLoc prev_loc =
      prev_class.is_defined() ? prev_class.definition_id : prev_class.decl_id;

  // TODO: Check that the generic parameter list agrees with the prior
  // declaration.

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
    prev_class.definition_id = new_class.definition_id;
    prev_class.scope_id = new_class.scope_id;
    prev_class.body_block_id = new_class.body_block_id;
    prev_class.adapt_id = new_class.adapt_id;
    prev_class.base_id = new_class.base_id;
    prev_class.object_repr_id = new_class.object_repr_id;
  }

  return true;
}

}  // namespace Carbon::Check
