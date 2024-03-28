// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/function.h"

#include "toolchain/check/subst.h"

namespace Carbon::Check {

// Returns true if there was an error in declaring the function, which will have
// previously been diagnosed.
static auto FunctionDeclHasError(Context& context, const SemIR::Function& fn)
    -> bool {
  if (fn.return_type_id == SemIR::TypeId::Error) {
    return true;
  }
  for (auto param_refs_id : {fn.implicit_param_refs_id, fn.param_refs_id}) {
    if (param_refs_id != SemIR::InstBlockId::Empty) {
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
    CARBON_DIAGNOSTIC(FunctionRedeclParamDiffers, Error,
                      "Function redeclaration differs at {0}parameter {1}.",
                      llvm::StringLiteral, int32_t);
    CARBON_DIAGNOSTIC(FunctionRedeclParamPrevious, Note,
                      "Previous declaration's corresponding {0}parameter here.",
                      llvm::StringLiteral);
    context.emitter()
        .Build(new_param_ref_id, FunctionRedeclParamDiffers, param_diag_label,
               param_index + 1)
        .Note(prev_param_ref_id, FunctionRedeclParamPrevious, param_diag_label)
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
  const auto new_param_ref_ids = context.inst_blocks().Get(new_param_refs_id);
  const auto prev_param_ref_ids = context.inst_blocks().Get(prev_param_refs_id);
  if (new_param_ref_ids.size() != prev_param_ref_ids.size()) {
    CARBON_DIAGNOSTIC(
        FunctionRedeclParamCountDiffers, Error,
        "Function redeclaration differs because of {0}parameter count of {1}.",
        llvm::StringLiteral, int32_t);
    CARBON_DIAGNOSTIC(FunctionRedeclParamCountPrevious, Note,
                      "Previously declared with {0}parameter count of {1}.",
                      llvm::StringLiteral, int32_t);
    context.emitter()
        .Build(new_decl_id, FunctionRedeclParamCountDiffers, param_diag_label,
               new_param_ref_ids.size())
        .Note(prev_decl_id, FunctionRedeclParamCountPrevious, param_diag_label,
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

// Returns false if the provided function declarations differ.
static auto CheckRedecl(Context& context, const SemIR::Function& new_function,
                        const SemIR::Function& prev_function,
                        Substitutions substitutions) -> bool {
  if (FunctionDeclHasError(context, new_function) ||
      FunctionDeclHasError(context, prev_function)) {
    return false;
  }
  if (!CheckRedeclParams(
          context, new_function.decl_id, new_function.implicit_param_refs_id,
          prev_function.decl_id, prev_function.implicit_param_refs_id,
          "implicit ", substitutions) ||
      !CheckRedeclParams(context, new_function.decl_id,
                         new_function.param_refs_id, prev_function.decl_id,
                         prev_function.param_refs_id, "", substitutions)) {
    return false;
  }
  auto prev_return_type_id =
      prev_function.return_type_id.is_valid()
          ? SubstType(context, prev_function.return_type_id, substitutions)
          : SemIR::TypeId::Invalid;
  if (new_function.return_type_id != prev_return_type_id) {
    CARBON_DIAGNOSTIC(
        FunctionRedeclReturnTypeDiffers, Error,
        "Function redeclaration differs because return type is `{0}`.",
        SemIR::TypeId);
    CARBON_DIAGNOSTIC(
        FunctionRedeclReturnTypeDiffersNoReturn, Error,
        "Function redeclaration differs because no return type is provided.");
    auto diag =
        new_function.return_type_id.is_valid()
            ? context.emitter().Build(new_function.decl_id,
                                      FunctionRedeclReturnTypeDiffers,
                                      new_function.return_type_id)
            : context.emitter().Build(new_function.decl_id,
                                      FunctionRedeclReturnTypeDiffersNoReturn);
    if (prev_return_type_id.is_valid()) {
      CARBON_DIAGNOSTIC(FunctionRedeclReturnTypePrevious, Note,
                        "Previously declared with return type `{0}`.",
                        SemIR::TypeId);
      diag.Note(prev_function.decl_id, FunctionRedeclReturnTypePrevious,
                prev_return_type_id);
    } else {
      CARBON_DIAGNOSTIC(FunctionRedeclReturnTypePreviousNoReturn, Note,
                        "Previously declared with no return type.");
      diag.Note(prev_function.decl_id,
                FunctionRedeclReturnTypePreviousNoReturn);
    }
    diag.Emit();
    return false;
  }

  return true;
}

auto CheckFunctionTypeMatches(Context& context,
                              SemIR::FunctionId new_function_id,
                              SemIR::FunctionId prev_function_id,
                              Substitutions substitutions) -> bool {
  return CheckRedecl(context, context.functions().Get(new_function_id),
                     context.functions().Get(prev_function_id), substitutions);
}

// Checks to see if a structurally valid redeclaration is allowed in context.
// These all still merge.
static auto CheckIsAllowedRedecl(Context& context, SemIR::LocId loc_id,
                                 SemIR::Function& new_function,
                                 bool new_is_definition,
                                 SemIR::Function& prev_function,
                                 bool prev_is_import) -> void {
  CARBON_DIAGNOSTIC(FunctionPreviousDecl, Note, "Previously declared here.");
  if (prev_is_import) {
    // TODO: Allow non-extern declarations in the same library.
    if (!new_function.is_extern && !prev_function.is_extern) {
      CARBON_DIAGNOSTIC(
          FunctionNonExternRedecl, Error,
          "Only one library can declare function {0} without `extern`.",
          SemIR::NameId);
      context.emitter()
          .Build(loc_id, FunctionNonExternRedecl, prev_function.name_id)
          .Note(prev_function.decl_id, FunctionPreviousDecl)
          .Emit();
      return;
    }
  } else {
    if (!new_is_definition) {
      CARBON_DIAGNOSTIC(FunctionRedecl, Error,
                        "Redundant redeclaration of function {0}.",
                        SemIR::NameId);
      context.emitter()
          .Build(loc_id, FunctionRedecl, prev_function.name_id)
          .Note(prev_function.decl_id, FunctionPreviousDecl)
          .Emit();
      return;
    }
    if (prev_function.definition_id.is_valid()) {
      CARBON_DIAGNOSTIC(FunctionRedefinition, Error,
                        "Redefinition of function {0}.", SemIR::NameId);
      CARBON_DIAGNOSTIC(FunctionPreviousDefinition, Note,
                        "Previously defined here.");
      context.emitter()
          .Build(loc_id, FunctionRedefinition, prev_function.name_id)
          .Note(prev_function.definition_id, FunctionPreviousDefinition)
          .Emit();
      return;
    }
    // `extern` definitions are prevented in handle_function.cpp; this is only
    // checking for a non-`extern` definition after an `extern` declaration.
    if (prev_function.is_extern) {
      CARBON_DIAGNOSTIC(FunctionDefiningExtern, Error,
                        "Redeclaring `extern` function `{0}` as non-`extern`.",
                        SemIR::NameId);
      CARBON_DIAGNOSTIC(FunctionPreviousExternDecl, Note,
                        "Previously declared `extern` here.");
      context.emitter()
          .Build(loc_id, FunctionDefiningExtern, prev_function.name_id)
          .Note(prev_function.decl_id, FunctionPreviousExternDecl)
          .Emit();
      return;
    }
  }
}

// TODO: Detect conflicting cross-file declarations, as well as uses of imported
// declarations followed by a redeclaration.
auto MergeFunctionRedecl(Context& context, SemIR::LocId loc_id,
                         SemIR::Function& new_function, bool new_is_definition,
                         SemIR::FunctionId prev_function_id,
                         bool prev_is_import) -> bool {
  auto& prev_function = context.functions().Get(prev_function_id);

  if (!CheckRedecl(context, new_function, prev_function, {})) {
    return false;
  }

  CheckIsAllowedRedecl(context, loc_id, new_function, new_is_definition,
                       prev_function, prev_is_import);

  if (new_is_definition) {
    // Track the signature from the definition, so that IDs in the body
    // match IDs in the signature.
    prev_function.definition_id = new_function.definition_id;
    prev_function.implicit_param_refs_id = new_function.implicit_param_refs_id;
    prev_function.param_refs_id = new_function.param_refs_id;
    prev_function.return_type_id = new_function.return_type_id;
    prev_function.return_slot_id = new_function.return_slot_id;
  }
  if (!new_function.is_extern) {
    prev_function.is_extern = false;
  }
  return true;
}

}  // namespace Carbon::Check
