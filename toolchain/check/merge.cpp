// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/merge.h"

#include "toolchain/base/kind_switch.h"
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
                          RedeclInfo prev_decl, SemIR::ImportIRId import_ir_id)
    -> void {
  if (!import_ir_id.is_valid()) {
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

auto ReplacePrevInstForMerge(Context& context, SemIR::NameScopeId scope_id,
                             SemIR::NameId name_id, SemIR::InstId new_inst_id)
    -> void {
  auto& names = context.name_scopes().Get(scope_id).names;
  auto it = names.find(name_id);
  if (it != names.end()) {
    it->second = new_inst_id;
  }
}

// Returns true if there was an error in declaring the entity, which will have
// previously been diagnosed.
static auto EntityHasParamError(Context& context, const DeclParams& info)
    -> bool {
  for (auto param_refs_id : {info.implicit_param_refs_id, info.param_refs_id}) {
    if (param_refs_id.is_valid() &&
        param_refs_id != SemIR::InstBlockId::Empty) {
      for (auto param_id : context.inst_blocks().Get(param_refs_id)) {
        if (context.insts().Get(param_id).type_id() == SemIR::TypeId::Error) {
          return true;
        }
      }
    }
  }
  return false;
}

// Returns false if a param differs for a redeclaration. The caller is expected
// to provide a diagnostic.
static auto CheckRedeclParam(Context& context,
                             llvm::StringLiteral param_diag_label,
                             int32_t param_index,
                             SemIR::InstId new_param_ref_id,
                             SemIR::InstId prev_param_ref_id,
                             Substitutions substitutions) -> bool {
  // TODO: Consider differentiating between type and name mistakes. For now,
  // taking the simpler approach because I also think we may want to refactor
  // params.
  auto diagnose = [&]() {
    CARBON_DIAGNOSTIC(RedeclParamDiffers, Error,
                      "Redeclaration differs at {0}parameter {1}.",
                      llvm::StringLiteral, int32_t);
    CARBON_DIAGNOSTIC(RedeclParamPrevious, Note,
                      "Previous declaration's corresponding {0}parameter here.",
                      llvm::StringLiteral);
    context.emitter()
        .Build(new_param_ref_id, RedeclParamDiffers, param_diag_label,
               param_index + 1)
        .Note(prev_param_ref_id, RedeclParamPrevious, param_diag_label)
        .Emit();
  };

  auto new_param_ref = context.insts().Get(new_param_ref_id);
  auto prev_param_ref = context.insts().Get(prev_param_ref_id);
  if (new_param_ref.kind() != prev_param_ref.kind() ||
      new_param_ref.type_id() !=
          SubstType(context, prev_param_ref.type_id(), substitutions)) {
    diagnose();
    return false;
  }

  if (new_param_ref.Is<SemIR::AddrPattern>()) {
    new_param_ref =
        context.insts().Get(new_param_ref.As<SemIR::AddrPattern>().inner_id);
    prev_param_ref =
        context.insts().Get(prev_param_ref.As<SemIR::AddrPattern>().inner_id);
    if (new_param_ref.kind() != prev_param_ref.kind()) {
      diagnose();
      return false;
    }
  }

  if (new_param_ref.Is<SemIR::AnyBindName>()) {
    new_param_ref =
        context.insts().Get(new_param_ref.As<SemIR::AnyBindName>().value_id);
    prev_param_ref =
        context.insts().Get(prev_param_ref.As<SemIR::AnyBindName>().value_id);
  }

  auto new_param = new_param_ref.As<SemIR::Param>();
  auto prev_param = prev_param_ref.As<SemIR::Param>();
  if (new_param.name_id != prev_param.name_id) {
    diagnose();
    return false;
  }

  return true;
}

// Returns false if the param refs differ for a redeclaration.
static auto CheckRedeclParams(Context& context, SemIR::InstId new_decl_id,
                              SemIR::InstBlockId new_param_refs_id,
                              SemIR::InstId prev_decl_id,
                              SemIR::InstBlockId prev_param_refs_id,
                              llvm::StringLiteral param_diag_label,
                              Substitutions substitutions) -> bool {
  // This will often occur for empty params.
  if (new_param_refs_id == prev_param_refs_id) {
    return true;
  }

  // If exactly one of the parameter lists was present, they differ.
  if (new_param_refs_id.is_valid() != prev_param_refs_id.is_valid()) {
    CARBON_DIAGNOSTIC(RedeclParamListDiffers, Error,
                      "Redeclaration differs because of {1}{0}parameter list.",
                      llvm::StringLiteral, llvm::StringLiteral);
    CARBON_DIAGNOSTIC(RedeclParamListPrevious, Note,
                      "Previously declared with{1} {0}parameter list.",
                      llvm::StringLiteral, llvm::StringLiteral);
    context.emitter()
        .Build(
            new_decl_id, RedeclParamListDiffers, param_diag_label,
            new_param_refs_id.is_valid() ? llvm::StringLiteral("") : "missing ")
        .Note(prev_decl_id, RedeclParamListPrevious, param_diag_label,
              prev_param_refs_id.is_valid() ? llvm::StringLiteral("") : "out")
        .Emit();
    return false;
  }

  CARBON_CHECK(new_param_refs_id.is_valid() && prev_param_refs_id.is_valid());
  const auto new_param_ref_ids = context.inst_blocks().Get(new_param_refs_id);
  const auto prev_param_ref_ids = context.inst_blocks().Get(prev_param_refs_id);
  if (new_param_ref_ids.size() != prev_param_ref_ids.size()) {
    CARBON_DIAGNOSTIC(
        RedeclParamCountDiffers, Error,
        "Redeclaration differs because of {0}parameter count of {1}.",
        llvm::StringLiteral, int32_t);
    CARBON_DIAGNOSTIC(RedeclParamCountPrevious, Note,
                      "Previously declared with {0}parameter count of {1}.",
                      llvm::StringLiteral, int32_t);
    context.emitter()
        .Build(new_decl_id, RedeclParamCountDiffers, param_diag_label,
               new_param_ref_ids.size())
        .Note(prev_decl_id, RedeclParamCountPrevious, param_diag_label,
              prev_param_ref_ids.size())
        .Emit();
    return false;
  }
  for (auto [index, new_param_ref_id, prev_param_ref_id] :
       llvm::enumerate(new_param_ref_ids, prev_param_ref_ids)) {
    if (!CheckRedeclParam(context, param_diag_label, index, new_param_ref_id,
                          prev_param_ref_id, substitutions)) {
      return false;
    }
  }
  return true;
}

auto CheckRedeclParamsMatch(Context& context, const DeclParams& new_entity,
                            const DeclParams& prev_entity,
                            Substitutions substitutions) -> bool {
  if (EntityHasParamError(context, new_entity) ||
      EntityHasParamError(context, prev_entity)) {
    return false;
  }
  if (!CheckRedeclParams(context, new_entity.decl_id,
                         new_entity.implicit_param_refs_id, prev_entity.decl_id,
                         prev_entity.implicit_param_refs_id, "implicit ",
                         substitutions) ||
      !CheckRedeclParams(context, new_entity.decl_id, new_entity.param_refs_id,
                         prev_entity.decl_id, prev_entity.param_refs_id, "",
                         substitutions)) {
    return false;
  }
  return true;
}

}  // namespace Carbon::Check
