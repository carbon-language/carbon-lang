// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/function.h"

#include "common/find.h"
#include "toolchain/check/merge.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/pattern.h"

namespace Carbon::Check {

auto FindSelfPattern(Context& context,
                     SemIR::InstBlockId implicit_param_patterns_id)
    -> SemIR::InstId {
  auto implicit_param_patterns =
      context.inst_blocks().GetOrEmpty(implicit_param_patterns_id);
  return FindIfOrNone(implicit_param_patterns, [&](auto implicit_param_id) {
    return SemIR::IsSelfPattern(context.sem_ir(), implicit_param_id);
  });
}

auto CheckFunctionReturnTypeMatches(Context& context,
                                    const SemIR::Function& new_function,
                                    const SemIR::Function& prev_function,
                                    SemIR::SpecificId prev_specific_id,
                                    bool diagnose) -> bool {
  // TODO: Pass a specific ID for `prev_function` instead of substitutions and
  // use it here.
  auto new_return_type_id =
      new_function.GetDeclaredReturnType(context.sem_ir());
  auto prev_return_type_id =
      prev_function.GetDeclaredReturnType(context.sem_ir(), prev_specific_id);
  if (new_return_type_id == SemIR::ErrorInst::TypeId ||
      prev_return_type_id == SemIR::ErrorInst::TypeId) {
    return false;
  }
  if (!context.types().AreEqualAcrossDeclarations(new_return_type_id,
                                                  prev_return_type_id)) {
    if (!diagnose) {
      return false;
    }

    CARBON_DIAGNOSTIC(
        FunctionRedeclReturnTypeDiffers, Error,
        "function redeclaration differs because return type is {0}",
        SemIR::TypeId);
    CARBON_DIAGNOSTIC(
        FunctionRedeclReturnTypeDiffersNoReturn, Error,
        "function redeclaration differs because no return type is provided");
    auto diag =
        new_return_type_id.has_value()
            ? context.emitter().Build(new_function.latest_decl_id(),
                                      FunctionRedeclReturnTypeDiffers,
                                      new_return_type_id)
            : context.emitter().Build(new_function.latest_decl_id(),
                                      FunctionRedeclReturnTypeDiffersNoReturn);
    if (prev_return_type_id.has_value()) {
      CARBON_DIAGNOSTIC(FunctionRedeclReturnTypePrevious, Note,
                        "previously declared with return type {0}",
                        SemIR::TypeId);
      diag.Note(prev_function.latest_decl_id(),
                FunctionRedeclReturnTypePrevious, prev_return_type_id);
    } else {
      CARBON_DIAGNOSTIC(FunctionRedeclReturnTypePreviousNoReturn, Note,
                        "previously declared with no return type");
      diag.Note(prev_function.latest_decl_id(),
                FunctionRedeclReturnTypePreviousNoReturn);
    }
    diag.Emit();
    return false;
  }

  return true;
}

auto CheckFunctionTypeMatches(Context& context,
                              const SemIR::Function& new_function,
                              const SemIR::Function& prev_function,
                              SemIR::SpecificId prev_specific_id,
                              bool check_syntax, bool check_self, bool diagnose)
    -> bool {
  if (!CheckRedeclParamsMatch(context, DeclParams(new_function),
                              DeclParams(prev_function), prev_specific_id,
                              diagnose, check_syntax, check_self)) {
    return false;
  }
  return CheckFunctionReturnTypeMatches(context, new_function, prev_function,
                                        prev_specific_id, diagnose);
}

auto CheckFunctionReturnType(Context& context, SemIR::LocId loc_id,
                             const SemIR::Function& function,
                             SemIR::SpecificId specific_id)
    -> SemIR::ReturnTypeInfo {
  auto return_info = SemIR::ReturnTypeInfo::ForFunction(context.sem_ir(),
                                                        function, specific_id);

  // If we couldn't determine the return information due to the return type
  // being incomplete, try to complete it now.
  if (return_info.init_repr.kind == SemIR::InitRepr::Incomplete) {
    auto diagnose_incomplete_return_type = [&] {
      CARBON_DIAGNOSTIC(IncompleteTypeInFunctionReturnType, Error,
                        "function returns incomplete type {0}", SemIR::TypeId);
      return context.emitter().Build(loc_id, IncompleteTypeInFunctionReturnType,
                                     return_info.type_id);
    };
    auto diagnose_abstract_return_type = [&] {
      CARBON_DIAGNOSTIC(AbstractTypeInFunctionReturnType, Error,
                        "function returns abstract type {0}", SemIR::TypeId);
      return context.emitter().Build(loc_id, AbstractTypeInFunctionReturnType,
                                     return_info.type_id);
    };

    // TODO: Consider suppressing the diagnostic if we've already diagnosed a
    // definition or call to this function.
    if (RequireConcreteType(context, return_info.type_id, loc_id,
                            diagnose_incomplete_return_type,
                            diagnose_abstract_return_type)) {
      return_info = SemIR::ReturnTypeInfo::ForFunction(context.sem_ir(),
                                                       function, specific_id);
    }
  }

  return return_info;
}

auto CheckFunctionDefinitionSignature(Context& context,
                                      SemIR::FunctionId function_id) -> void {
  auto& function = context.functions().Get(function_id);

  auto params_to_complete =
      context.inst_blocks().GetOrEmpty(function.call_params_id);

  // Check the return type is complete.
  if (function.return_slot_pattern_id.has_value()) {
    CheckFunctionReturnType(context,
                            SemIR::LocId(function.return_slot_pattern_id),
                            function, SemIR::SpecificId::None);
    // Don't re-check the return type below.
    params_to_complete = params_to_complete.drop_back();
  }

  // Check the parameter types are complete.
  for (auto param_ref_id : params_to_complete) {
    if (param_ref_id == SemIR::ErrorInst::InstId) {
      continue;
    }

    // The parameter types need to be complete.
    RequireCompleteType(
        context, context.insts().GetAs<SemIR::AnyParam>(param_ref_id).type_id,
        SemIR::LocId(param_ref_id), [&] {
          CARBON_DIAGNOSTIC(
              IncompleteTypeInFunctionParam, Error,
              "parameter has incomplete type {0} in function definition",
              TypeOfInstId);
          return context.emitter().Build(
              param_ref_id, IncompleteTypeInFunctionParam, param_ref_id);
        });
  }
}

}  // namespace Carbon::Check
