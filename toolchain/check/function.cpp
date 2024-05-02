// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/function.h"

#include "toolchain/check/merge.h"
#include "toolchain/check/subst.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Returns false if the provided function declarations differ.
static auto CheckRedecl(Context& context, const SemIR::Function& new_function,
                        const SemIR::Function& prev_function,
                        Substitutions substitutions) -> bool {
  if (!CheckRedeclParamsMatch(context, EntityInfo(new_function),
                              EntityInfo(prev_function), substitutions)) {
    return false;
  }

  if (new_function.return_type_id == SemIR::TypeId::Error ||
      prev_function.return_type_id == SemIR::TypeId::Error) {
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

// Returns the return slot usage for a function given the computed usage for two
// different declarations of the function.
static auto MergeReturnSlot(SemIR::Function::ReturnSlot a,
                            SemIR::Function::ReturnSlot b)
    -> SemIR::Function::ReturnSlot {
  if (a == SemIR::Function::ReturnSlot::NotComputed) {
    return b;
  }
  if (b == SemIR::Function::ReturnSlot::NotComputed) {
    return a;
  }
  if (a == SemIR::Function::ReturnSlot::Error) {
    return b;
  }
  if (b == SemIR::Function::ReturnSlot::Error) {
    return a;
  }
  CARBON_CHECK(a == b)
      << "Different return slot usage computed for the same function.";
  return a;
}

auto MergeFunctionRedecl(Context& context, SemIRLoc new_loc,
                         SemIR::Function& new_function, bool new_is_import,
                         bool new_is_definition,
                         SemIR::FunctionId prev_function_id,
                         SemIR::ImportIRInstId prev_import_ir_inst_id) -> bool {
  auto& prev_function = context.functions().Get(prev_function_id);

  if (!CheckRedecl(context, new_function, prev_function, {})) {
    return false;
  }

  CheckIsAllowedRedecl(context, Lex::TokenKind::Fn, prev_function.name_id,
                       {.loc = new_loc,
                        .is_definition = new_is_definition,
                        .is_extern = new_function.is_extern},
                       {.loc = prev_function.definition_id.is_valid()
                                   ? prev_function.definition_id
                                   : prev_function.decl_id,
                        .is_definition = prev_function.definition_id.is_valid(),
                        .is_extern = prev_function.is_extern},
                       prev_import_ir_inst_id);

  if (new_is_definition) {
    // Track the signature from the definition, so that IDs in the body
    // match IDs in the signature.
    prev_function.definition_id = new_function.definition_id;
    prev_function.implicit_param_refs_id = new_function.implicit_param_refs_id;
    prev_function.param_refs_id = new_function.param_refs_id;
    prev_function.return_type_id = new_function.return_type_id;
    prev_function.return_storage_id = new_function.return_storage_id;
  }
  // The new function might have return slot information if it was imported.
  prev_function.return_slot =
      MergeReturnSlot(prev_function.return_slot, new_function.return_slot);
  if ((prev_import_ir_inst_id.is_valid() && !new_is_import) ||
      (prev_function.is_extern && !new_function.is_extern)) {
    prev_function.is_extern = new_function.is_extern;
    prev_function.decl_id = new_function.decl_id;
    ReplacePrevInstForMerge(context, prev_function.enclosing_scope_id,
                            prev_function.name_id, new_function.decl_id);
  }
  return true;
}

auto CheckFunctionReturnType(Context& context, SemIRLoc loc,
                             SemIR::Function& function) -> void {
  // If we have already checked the return type, we have nothing to do.
  if (function.return_slot != SemIR::Function::ReturnSlot::NotComputed) {
    return;
  }

  if (!function.return_type_id.is_valid()) {
    // Implicit `-> ()` has no return slot.
    function.return_slot = SemIR::Function::ReturnSlot::Absent;
    return;
  }

  // Check the return type is complete. Only diagnose incompleteness if we've
  // not already done so.
  auto diagnose_incomplete_return_type = [&] {
    CARBON_DIAGNOSTIC(IncompleteTypeInFunctionReturnType, Error,
                      "Function returns incomplete type `{0}`.", SemIR::TypeId);
    return context.emitter().Build(loc, IncompleteTypeInFunctionReturnType,
                                   function.return_type_id);
  };
  if (!context.TryToCompleteType(
          function.return_type_id,
          function.return_slot == SemIR::Function::ReturnSlot::Error
              ? std::nullopt
              : std::optional(diagnose_incomplete_return_type))) {
    function.return_slot = SemIR::Function::ReturnSlot::Error;
  } else if (SemIR::GetInitRepr(context.sem_ir(), function.return_type_id)
                 .has_return_slot()) {
    function.return_slot = SemIR::Function::ReturnSlot::Present;
  } else {
    function.return_slot = SemIR::Function::ReturnSlot::Absent;
  }
}

}  // namespace Carbon::Check
