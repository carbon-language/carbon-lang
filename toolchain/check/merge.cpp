// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/merge.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/class.h"
#include "toolchain/check/function.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

CARBON_DIAGNOSTIC(RedeclPrevDecl, Note, "Previously declared here.");

// Diagnoses a redeclaration which is redundant.
static auto DiagnoseRedundant(Context& context, Lex::TokenKind decl_kind,
                              SemIR::NameId name_id, SemIRLoc new_loc,
                              SemIRLoc prev_loc) {
  CARBON_DIAGNOSTIC(RedeclRedundant, Error,
                    "Redeclaration of `{0} {1}` is redundant.", Lex::TokenKind,
                    SemIR::NameId);
  context.emitter()
      .Build(new_loc, RedeclRedundant, decl_kind, name_id)
      .Note(prev_loc, RedeclPrevDecl)
      .Emit();
}

// Diagnoses a redefinition.
static auto DiagnoseRedef(Context& context, Lex::TokenKind decl_kind,
                          SemIR::NameId name_id, SemIRLoc new_loc,
                          SemIRLoc prev_loc) {
  CARBON_DIAGNOSTIC(RedeclRedef, Error, "Redefinition of `{0} {1}`.",
                    Lex::TokenKind, SemIR::NameId);
  CARBON_DIAGNOSTIC(RedeclPrevDef, Note, "Previously defined here.");
  context.emitter()
      .Build(new_loc, RedeclRedef, decl_kind, name_id)
      .Note(prev_loc, RedeclPrevDef)
      .Emit();
}

// Diagnoses an `extern` versus non-`extern` mismatch.
static auto DiagnoseExternMismatch(Context& context, Lex::TokenKind decl_kind,
                                   SemIR::NameId name_id, SemIRLoc new_loc,
                                   SemIRLoc prev_loc) {
  CARBON_DIAGNOSTIC(RedeclExternMismatch, Error,
                    "Redeclarations of `{0} {1}` in the same library must "
                    "match use of `extern`.",
                    Lex::TokenKind, SemIR::NameId);
  context.emitter()
      .Build(new_loc, RedeclExternMismatch, decl_kind, name_id)
      .Note(prev_loc, RedeclPrevDecl)
      .Emit();
}

// Diagnoses when multiple non-`extern` declarations are found.
static auto DiagnoseNonExtern(Context& context, Lex::TokenKind decl_kind,
                              SemIR::NameId name_id, SemIRLoc new_loc,
                              SemIRLoc prev_loc) {
  CARBON_DIAGNOSTIC(RedeclNonExtern, Error,
                    "Only one library can declare `{0} {1}` without `extern`.",
                    Lex::TokenKind, SemIR::NameId);
  context.emitter()
      .Build(new_loc, RedeclNonExtern, decl_kind, name_id)
      .Note(prev_loc, RedeclPrevDecl)
      .Emit();
}

// Checks to see if a structurally valid redeclaration is allowed in context.
// These all still merge.
auto CheckIsAllowedRedecl(Context& context, Lex::TokenKind decl_kind,
                          SemIR::NameId name_id, RedeclInfo new_decl,
                          RedeclInfo prev_decl,
                          SemIR::ImportIRInstId prev_import_ir_inst_id)
    -> void {
  if (!prev_import_ir_inst_id.is_valid()) {
    // Check for disallowed redeclarations in the same file.
    if (!new_decl.is_definition) {
      DiagnoseRedundant(context, decl_kind, name_id, new_decl.loc,
                        prev_decl.loc);
      return;
    }
    if (prev_decl.is_definition) {
      DiagnoseRedef(context, decl_kind, name_id, new_decl.loc, prev_decl.loc);
      return;
    }
    // `extern` definitions are prevented at creation; this is only
    // checking for a non-`extern` definition after an `extern` declaration.
    if (prev_decl.is_extern) {
      DiagnoseExternMismatch(context, decl_kind, name_id, new_decl.loc,
                             prev_decl.loc);
      return;
    }
    return;
  }

  auto import_ir_id =
      context.import_ir_insts().Get(prev_import_ir_inst_id).ir_id;
  if (import_ir_id == SemIR::ImportIRId::ApiForImpl) {
    // Check for disallowed redeclarations in the same library. Note that a
    // forward declaration in the impl is allowed.
    if (prev_decl.is_definition) {
      if (new_decl.is_definition) {
        DiagnoseRedef(context, decl_kind, name_id, new_decl.loc, prev_decl.loc);
      } else {
        DiagnoseRedundant(context, decl_kind, name_id, new_decl.loc,
                          prev_decl.loc);
      }
      return;
    }
    if (prev_decl.is_extern != new_decl.is_extern) {
      DiagnoseExternMismatch(context, decl_kind, name_id, new_decl.loc,
                             prev_decl.loc);
      return;
    }
    return;
  }

  // Check for disallowed redeclarations cross-library.
  if (!new_decl.is_extern && !prev_decl.is_extern) {
    DiagnoseNonExtern(context, decl_kind, name_id, new_decl.loc, prev_decl.loc);
    return;
  }
}

auto ResolvePrevInstForMerge(Context& context, Parse::NodeId node_id,
                             SemIR::InstId prev_inst_id) -> InstForMerge {
  auto prev_inst = context.insts().Get(prev_inst_id);
  auto import_ref = prev_inst.TryAs<SemIR::AnyImportRef>();
  // If not imported, use the instruction directly.
  if (!import_ref) {
    return {.inst = prev_inst,
            .import_ir_inst_id = SemIR::ImportIRInstId::Invalid};
  }

  // If the import ref was previously used, print a diagnostic.
  if (auto import_ref_used = prev_inst.TryAs<SemIR::ImportRefUsed>()) {
    CARBON_DIAGNOSTIC(
        RedeclOfUsedImport, Error,
        "Redeclaration of imported entity that was previously used.");
    CARBON_DIAGNOSTIC(UsedImportLoc, Note, "Import used here.");
    context.emitter()
        .Build(node_id, RedeclOfUsedImport)
        .Note(import_ref_used->used_id, UsedImportLoc)
        .Emit();
  }

  // Follow the import ref.
  return {.inst = context.insts().Get(
              context.constant_values().Get(prev_inst_id).inst_id()),
          .import_ir_inst_id = import_ref->import_ir_inst_id};
}

// Returns the instruction to consider when merging the given inst_id. Returns
// nullopt if merging is infeasible and no diagnostic should be printed.
static auto ResolveMergeableInst(Context& context, SemIR::InstId inst_id)
    -> std::optional<InstForMerge> {
  auto inst = context.insts().Get(inst_id);
  switch (inst.kind()) {
    case SemIR::ImportRefUnloaded::Kind:
      // Load before merging.
      LoadImportRef(context, inst_id, SemIR::LocId::Invalid);
      break;

    case SemIR::ImportRefLoaded::Kind:
    case SemIR::ImportRefUsed::Kind:
      // Already loaded.
      break;

    case SemIR::Namespace::Kind:
      // Return back the namespace directly.
      return {
          {.inst = inst, .import_ir_inst_id = SemIR::ImportIRInstId::Invalid}};

    default:
      CARBON_FATAL() << "Unexpected inst kind passed to ResolveMergeableInst: "
                     << inst;
  }

  auto const_id = context.constant_values().Get(inst_id);
  // TODO: Function and type declarations are constant, but `var` declarations
  // are non-constant and should still merge.
  if (!const_id.is_constant()) {
    return std::nullopt;
  }
  return {
      {.inst = context.insts().Get(const_id.inst_id()),
       .import_ir_inst_id = inst.As<SemIR::AnyImportRef>().import_ir_inst_id}};
}

auto ReplacePrevInstForMerge(Context& context, SemIR::NameScopeId scope_id,
                             SemIR::NameId name_id, SemIR::InstId new_inst_id)
    -> void {
  auto& names = context.name_scopes().Get(scope_id).names;
  auto it = names.find(name_id);
  if (it != names.end()) {
    it->second = new_inst_id;
  }
}

// TODO: On successful merges, this may need to "spoil" new_inst_id in order to
// prevent it from being emitted in lowering.
auto MergeImportRef(Context& context, SemIR::InstId new_inst_id,
                    SemIR::InstId prev_inst_id) -> void {
  auto new_inst = ResolveMergeableInst(context, new_inst_id);
  auto prev_inst = ResolveMergeableInst(context, prev_inst_id);
  if (!new_inst || !prev_inst) {
    // TODO: Once `var` declarations get an associated instruction for handling,
    // it might be more appropriate to return without diagnosing here, to handle
    // invalid declarations.
    context.DiagnoseDuplicateName(new_inst_id, prev_inst_id);
    return;
  }

  if (new_inst->inst.kind() != prev_inst->inst.kind()) {
    context.DiagnoseDuplicateName(new_inst_id, prev_inst_id);
    return;
  }

  CARBON_KIND_SWITCH(new_inst->inst) {
    case CARBON_KIND(SemIR::FunctionDecl new_decl): {
      auto prev_decl = prev_inst->inst.TryAs<SemIR::FunctionDecl>();
      if (!prev_decl) {
        break;
      }

      auto new_fn = context.functions().Get(new_decl.function_id);
      MergeFunctionRedecl(context, new_inst_id, new_fn,
                          /*new_is_import=*/true,
                          /*new_is_definition=*/false, prev_decl->function_id,
                          prev_inst->import_ir_inst_id);
      return;
    }
    case CARBON_KIND(SemIR::ClassType new_type): {
      auto prev_type = prev_inst->inst.TryAs<SemIR::ClassType>();
      if (!prev_type) {
        break;
      }

      auto new_class = context.classes().Get(new_type.class_id);
      // TODO: Fix new_is_extern and prev_is_extern.
      MergeClassRedecl(context, new_inst_id, new_class,
                       /*new_is_import=*/true, new_class.is_defined(),
                       /*new_is_extern=*/false, prev_type->class_id,
                       /*prev_is_extern=*/false, prev_inst->import_ir_inst_id);
      return;
    }
    default:
      context.TODO(new_inst_id, llvm::formatv("Merging {0} not yet supported.",
                                              new_inst->inst.kind()));
      return;
  }
}

}  // namespace Carbon::Check
