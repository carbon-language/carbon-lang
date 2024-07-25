// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/function.h"

#include "toolchain/check/merge.h"
#include "toolchain/check/subst.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

auto CheckFunctionTypeMatches(Context& context,
                              const SemIR::Function& new_function,
                              const SemIR::Function& prev_function,
                              Substitutions substitutions, bool check_syntax)
    -> bool {
  if (!CheckRedeclParamsMatch(context, DeclParams(new_function),
                              DeclParams(prev_function), substitutions,
                              check_syntax)) {
    return false;
  }

  // TODO: Pass a specific ID for `prev_function` instead of substitutions and
  // use it here.
  auto new_return_type_id =
      new_function.GetDeclaredReturnType(context.sem_ir());
  auto prev_return_type_id = prev_function.GetDeclaredReturnType(
      context.sem_ir(), SemIR::GenericInstanceId::Invalid);
  if (new_return_type_id == SemIR::TypeId::Error ||
      prev_return_type_id == SemIR::TypeId::Error) {
    return false;
  }
  if (prev_return_type_id.is_valid()) {
    prev_return_type_id =
        SubstType(context, prev_return_type_id, substitutions);
  }
  if (!context.types().AreEqualAcrossDeclarations(new_return_type_id,
                                                  prev_return_type_id)) {
    CARBON_DIAGNOSTIC(
        FunctionRedeclReturnTypeDiffers, Error,
        "Function redeclaration differs because return type is `{0}`.",
        SemIR::TypeId);
    CARBON_DIAGNOSTIC(
        FunctionRedeclReturnTypeDiffersNoReturn, Error,
        "Function redeclaration differs because no return type is provided.");
    auto diag =
        new_return_type_id.is_valid()
            ? context.emitter().Build(new_function.decl_id,
                                      FunctionRedeclReturnTypeDiffers,
                                      new_return_type_id)
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

auto CheckFunctionReturnType(Context& context, SemIRLoc loc,
                             SemIR::Function& function,
                             SemIR::GenericInstanceId specific_id)
    -> SemIR::Function::ReturnInfo {
  auto return_info = function.GetReturnInfo(context.sem_ir(), specific_id);

  // If we couldn't determine the return information due to the return type
  // being incomplete, try to complete it now.
  if (return_info.return_slot == SemIR::Function::ReturnSlot::Incomplete) {
    auto diagnose_incomplete_return_type = [&] {
      CARBON_DIAGNOSTIC(IncompleteTypeInFunctionReturnType, Error,
                        "Function returns incomplete type `{0}`.",
                        SemIR::TypeId);
      return context.emitter().Build(loc, IncompleteTypeInFunctionReturnType,
                                     return_info.type_id);
    };

    // TODO: Consider suppressing the diagnostic if we've already diagnosed a
    // definition or call to this function.
    if (context.TryToCompleteType(return_info.type_id,
                                  diagnose_incomplete_return_type)) {
      return_info = function.GetReturnInfo(context.sem_ir(), specific_id);
    }
  }

  return return_info;
}

}  // namespace Carbon::Check
