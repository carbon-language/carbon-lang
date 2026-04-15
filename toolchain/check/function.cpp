// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/function.h"

#include "common/find.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/merge.h"
#include "toolchain/check/pattern.h"
#include "toolchain/check/pattern_match.h"
#include "toolchain/check/scope_stack.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
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

auto AddReturnPatterns(Context& context, SemIR::LocId loc_id,
                       Context::FormExpr form_expr) -> SemIR::InstBlockId {
  llvm::SmallVector<SemIR::InstId, 1> return_patterns;
  auto form_inst = context.insts().Get(
      context.constant_values().GetConstantInstId(form_expr.form_inst_id));
  CARBON_KIND_SWITCH(form_inst) {
    case SemIR::RefForm::Kind:
    case SemIR::ValueForm::Kind: {
      break;
    }
    case CARBON_KIND(SemIR::InitForm _): {
      auto pattern_type_id =
          GetPatternType(context, form_expr.type_component_id);
      auto out_param_id = AddPatternInst<SemIR::OutParamPattern>(
          context, SemIR::LocId(form_expr.form_inst_id),
          {.type_id = pattern_type_id,
           .pretty_name_id = SemIR::NameId::ReturnSlot});
      return_patterns.push_back(AddPatternInst<SemIR::ReturnSlotPattern>(
          context, SemIR::LocId(form_expr.form_inst_id),
          {.type_id = pattern_type_id,
           .subpattern_id = out_param_id,
           .type_inst_id = form_expr.type_component_inst_id}));
      break;
    }
    case SemIR::ErrorInst::Kind: {
      break;
    }
    case SemIR::SymbolicBinding::Kind:
      CARBON_CHECK(
          context.constant_values().Get(form_expr.form_inst_id).is_symbolic());
      context.TODO(loc_id, "Support symbolic return forms");
      break;
    default:
      CARBON_FATAL("unexpected inst kind: {0}", form_inst);
  }
  return context.inst_blocks().AddCanonical(return_patterns);
}

auto IsValidBuiltinDeclaration(Context& context,
                               const SemIR::Function& function,
                               SemIR::BuiltinFunctionKind builtin_kind)
    -> bool {
  if (!function.call_params_id.has_value()) {
    // For now, we have no builtins that support positional parameters.
    return false;
  }

  // Find the list of call parameters other than the implicit return slots.
  auto call_params =
      context.inst_blocks()
          .Get(function.call_params_id)
          .take_front(function.call_param_ranges.explicit_end().index);

  // Get the return type. This is `()` if none was specified.
  auto return_type_id = function.GetDeclaredReturnType(context.sem_ir());
  if (!return_type_id.has_value()) {
    return_type_id = GetTupleType(context, {});
  }

  return builtin_kind.IsValidType(context.sem_ir(), call_params,
                                  return_type_id);
}

namespace {
// Function signature fields for `MakeFunctionSignature`.
struct FunctionSignatureInsts {
  SemIR::InstBlockId decl_block_id = SemIR::InstBlockId::None;
  SemIR::InstBlockId pattern_block_id = SemIR::InstBlockId::None;
  SemIR::InstBlockId implicit_param_patterns_id = SemIR::InstBlockId::None;
  SemIR::InstBlockId param_patterns_id = SemIR::InstBlockId::None;
  SemIR::InstBlockId call_param_patterns_id = SemIR::InstBlockId::None;
  SemIR::InstBlockId call_params_id = SemIR::InstBlockId::None;
  SemIR::Function::CallParamIndexRanges call_param_ranges =
      SemIR::Function::CallParamIndexRanges::Empty;
  SemIR::TypeInstId return_type_inst_id = SemIR::TypeInstId::None;
  SemIR::InstId return_form_inst_id = SemIR::InstId::None;
  SemIR::InstBlockId return_patterns_id = SemIR::InstBlockId::None;
  SemIR::InstId self_param_id = SemIR::InstId::None;
};
}  // namespace

// Handles construction of the signature's parameter and return types.
static auto MakeFunctionSignature(Context& context, SemIR::LocId loc_id,
                                  const FunctionDeclArgs& args)
    -> FunctionSignatureInsts {
  FunctionSignatureInsts insts;

  StartFunctionSignature(context);

  // Build and add a `self: Self` or `ref self: Self` parameter if needed.
  if (args.self_type_id.has_value()) {
    context.full_pattern_stack().StartImplicitParamList();

    BeginSubpattern(context);
    auto self_type_region_id = ConsumeSubpatternExpr(
        context, context.types().GetTypeInstId(args.self_type_id));
    EndEmptySubpattern(context);

    insts.self_param_id = AddParamPattern(
        context, loc_id, SemIR::NameId::SelfValue, self_type_region_id,
        args.self_type_id, args.self_is_ref);
    insts.implicit_param_patterns_id =
        context.inst_blocks().Add({insts.self_param_id});

    context.full_pattern_stack().EndImplicitParamList();
  }

  // Build and add any explicit parameters. Whether these are references
  // or not is controlled by `args.params_are_refs`.
  context.full_pattern_stack().StartExplicitParamList();
  if (args.param_type_ids.empty()) {
    insts.param_patterns_id = SemIR::InstBlockId::Empty;
  } else {
    context.inst_block_stack().Push();
    for (auto param_type_id : args.param_type_ids) {
      BeginSubpattern(context);
      auto param_type_region_id = ConsumeSubpatternExpr(
          context, context.types().GetTypeInstId(param_type_id));
      EndEmptySubpattern(context);

      context.inst_block_stack().AddInstId(AddParamPattern(
          context, loc_id, SemIR::NameId::Underscore, param_type_region_id,
          param_type_id, /*is_ref=*/args.params_are_refs));
    }
    insts.param_patterns_id = context.inst_block_stack().Pop();
  }
  context.full_pattern_stack().EndExplicitParamList();

  // Build and add the return type. We always use an initializing form for now.
  if (args.return_type_id.has_value()) {
    auto return_form = ReturnExprAsForm(
        context, loc_id, context.types().GetTypeInstId(args.return_type_id));
    insts.return_type_inst_id = return_form.type_component_inst_id;
    insts.return_form_inst_id = return_form.form_inst_id;
    insts.return_patterns_id = AddReturnPatterns(context, loc_id, return_form);
  }

  auto match_results =
      CalleePatternMatch(context, insts.implicit_param_patterns_id,
                         insts.param_patterns_id, insts.return_patterns_id);
  insts.call_param_patterns_id = match_results.call_param_patterns_id;
  insts.call_params_id = match_results.call_params_id;
  insts.call_param_ranges = match_results.param_ranges;

  auto [pattern_block_id, decl_block_id] =
      FinishFunctionSignature(context, /*check_unused=*/false);
  insts.pattern_block_id = pattern_block_id;
  insts.decl_block_id = decl_block_id;

  return insts;
}

auto MakeGeneratedFunctionDecl(Context& context, SemIR::LocId loc_id,
                               const FunctionDeclArgs& args)
    -> std::pair<SemIR::InstId, SemIR::FunctionId> {
  auto insts = MakeFunctionSignature(context, loc_id, args);

  // Add the function declaration.
  auto [decl_id, function_id] = MakeFunctionDecl(
      context, loc_id, insts.decl_block_id, /*build_generic=*/false,
      /*is_definition=*/true,
      SemIR::Function{
          {
              .name_id = args.name_id,
              .parent_scope_id = args.parent_scope_id,
              .generic_id = SemIR::GenericId::None,
              .first_param_node_id = Parse::NodeId::None,
              .last_param_node_id = Parse::NodeId::None,
              .pattern_block_id = insts.pattern_block_id,
              .implicit_param_patterns_id = insts.implicit_param_patterns_id,
              .param_patterns_id = insts.param_patterns_id,
              .is_extern = false,
              .extern_library_id = SemIR::LibraryNameId::None,
              .non_owning_decl_id = SemIR::InstId::None,
              // Set by `MakeFunctionDecl`.
              .first_owning_decl_id = SemIR::InstId::None,
          },
          {
              .call_param_patterns_id = insts.call_param_patterns_id,
              .call_params_id = insts.call_params_id,
              .call_param_ranges = insts.call_param_ranges,
              .return_type_inst_id = insts.return_type_inst_id,
              .return_form_inst_id = insts.return_form_inst_id,
              .return_patterns_id = insts.return_patterns_id,
              .self_param_id = insts.self_param_id,
          }});
  context.generated().push_back(decl_id);

  return {decl_id, function_id};
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

// Checks that a function declaration's evaluation mode matches the previous
// declaration's evaluation mode. Returns `false` and optionally produces a
// diagnostic on mismatch.
static auto CheckFunctionEvaluationModeMatches(
    Context& context, const SemIR::Function& new_function,
    const SemIR::Function& prev_function, bool diagnose) -> bool {
  if (prev_function.evaluation_mode == new_function.evaluation_mode) {
    return true;
  }
  if (!diagnose) {
    return false;
  }
  auto eval_mode_index = [](SemIR::Function::EvaluationMode mode) {
    switch (mode) {
      case SemIR::Function::EvaluationMode::None:
        return 0;
      case SemIR::Function::EvaluationMode::Eval:
        return 1;
      case SemIR::Function::EvaluationMode::MustEval:
        return 2;
    }
  };
  auto prev_eval_mode_index = eval_mode_index(prev_function.evaluation_mode);
  auto new_eval_mode_index = eval_mode_index(new_function.evaluation_mode);
  CARBON_DIAGNOSTIC(
      FunctionRedeclEvaluationModeDiffers, Error,
      "function redeclaration differs because new function is "
      "{0:=-1:not `eval`|=-2:not `musteval`|=1:`eval`|=2:`musteval`}",
      Diagnostics::IntAsSelect);
  CARBON_DIAGNOSTIC(FunctionRedeclEvaluationModePrevious, Note,
                    "previously {0:<0:not |:}declared as "
                    "{0:=-1:`eval`|=-2:`musteval`|=1:`eval`|=2:`musteval`}",
                    Diagnostics::IntAsSelect);
  context.emitter()
      .Build(new_function.latest_decl_id(), FunctionRedeclEvaluationModeDiffers,
             new_eval_mode_index ? new_eval_mode_index : -prev_eval_mode_index)
      .Note(prev_function.latest_decl_id(),
            FunctionRedeclEvaluationModePrevious,
            prev_eval_mode_index ? prev_eval_mode_index : -new_eval_mode_index)
      .Emit();
  return false;
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
  if (!CheckFunctionReturnTypeMatches(context, new_function, prev_function,
                                      prev_specific_id, diagnose)) {
    return false;
  }
  if (!CheckFunctionEvaluationModeMatches(context, new_function, prev_function,
                                          diagnose)) {
    return false;
  }
  return true;
}

auto CheckFunctionReturnPatternType(Context& context, SemIR::LocId loc_id,
                                    SemIR::InstId return_pattern_id,
                                    SemIR::SpecificId specific_id)
    -> SemIR::TypeId {
  auto arg_type_id = SemIR::ExtractScrutineeType(
      context.sem_ir(), SemIR::GetTypeOfInstInSpecific(
                            context.sem_ir(), specific_id, return_pattern_id));
  auto init_repr = SemIR::InitRepr::ForType(context.sem_ir(), arg_type_id);
  if (!init_repr.is_valid()) {
    // TODO: Consider suppressing the diagnostics if we've already diagnosed a
    // definition or call to this function.
    if (!RequireConcreteType(
            context, arg_type_id, SemIR::LocId(return_pattern_id),
            [&](auto& builder) {
              CARBON_DIAGNOSTIC(IncompleteTypeInFunctionReturnType, Context,
                                "function returns incomplete type {0}",
                                SemIR::TypeId);
              builder.Context(loc_id, IncompleteTypeInFunctionReturnType,
                              arg_type_id);
            },
            [&](auto& builder) {
              CARBON_DIAGNOSTIC(AbstractTypeInFunctionReturnType, Context,
                                "function returns abstract type {0}",
                                SemIR::TypeId);
              builder.Context(loc_id, AbstractTypeInFunctionReturnType,
                              arg_type_id);
            })) {
      return SemIR::ErrorInst::TypeId;
    }
  }

  return arg_type_id;
}

auto CheckFunctionDefinitionSignature(Context& context,
                                      SemIR::FunctionId function_id) -> void {
  auto& function = context.functions().Get(function_id);

  auto params_to_complete =
      context.inst_blocks().GetOrEmpty(function.call_params_id);

  // The return parameter will be diagnosed after and differently from other
  // parameters.
  auto return_call_param = SemIR::InstId::None;
  if (!params_to_complete.empty() && function.return_patterns_id.has_value()) {
    return_call_param = params_to_complete.consume_back();
  }

  // Check the parameter types are complete.
  for (auto param_ref_id : params_to_complete) {
    if (param_ref_id == SemIR::ErrorInst::InstId) {
      continue;
    }

    // The parameter types need to be complete.
    RequireCompleteType(
        context, context.insts().GetAs<SemIR::AnyParam>(param_ref_id).type_id,
        SemIR::LocId(param_ref_id), [&](auto& builder) {
          CARBON_DIAGNOSTIC(
              IncompleteTypeInFunctionParam, Context,
              "parameter has incomplete type {0} in function definition",
              TypeOfInstId);
          builder.Context(param_ref_id, IncompleteTypeInFunctionParam,
                          param_ref_id);
        });
  }

  // Check the return type is complete.
  if (function.return_patterns_id.has_value()) {
    for (auto return_pattern_id :
         context.inst_blocks().Get(function.return_patterns_id)) {
      CheckFunctionReturnPatternType(context, SemIR::LocId(return_pattern_id),
                                     return_pattern_id,
                                     SemIR::SpecificId::None);
    }

    // `CheckFunctionReturnPatternType` should have diagnosed incomplete types,
    // so don't `RequireCompleteType` on the return type.
    if (return_call_param.has_value()) {
      // TODO: If the types are already checked for completeness then this does
      // nothing?
      TryToCompleteType(
          context,
          context.insts().GetAs<SemIR::AnyParam>(return_call_param).type_id,
          SemIR::LocId(return_call_param));
    }
  }
}

auto StartFunctionSignature(Context& context) -> void {
  context.scope_stack().PushForDeclName();
  context.inst_block_stack().Push();
  context.pattern_block_stack().Push();
  context.full_pattern_stack().PushParameterizedDecl();
}

auto FinishFunctionSignature(Context& context, bool check_unused)
    -> FinishFunctionSignatureResult {
  context.full_pattern_stack().PopFullPattern();
  auto pattern_block_id = context.pattern_block_stack().Pop();
  auto decl_block_id = context.inst_block_stack().Pop();
  context.scope_stack().Pop(check_unused);
  return {.pattern_block_id = pattern_block_id, .decl_block_id = decl_block_id};
}

auto MakeFunctionDecl(Context& context, SemIR::LocId loc_id,
                      SemIR::InstBlockId decl_block_id, bool build_generic,
                      bool is_definition, SemIR::Function function)
    -> std::pair<SemIR::InstId, SemIR::FunctionId> {
  CARBON_CHECK(!function.first_owning_decl_id.has_value());

  SemIR::FunctionDecl function_decl = {SemIR::TypeId::None,
                                       SemIR::FunctionId::None, decl_block_id};
  auto decl_id = AddPlaceholderInstInNoBlock(
      context, SemIR::LocIdAndInst::RuntimeVerified(context.sem_ir(), loc_id,
                                                    function_decl));
  function.first_owning_decl_id = decl_id;
  if (is_definition) {
    function.definition_id = decl_id;
  }

  if (build_generic) {
    function.generic_id = BuildGenericDecl(context, decl_id);
  }

  // Create the `Function` object.
  function_decl.function_id = context.functions().Add(std::move(function));
  function_decl.type_id =
      GetFunctionType(context, function_decl.function_id,
                      build_generic ? context.scope_stack().PeekSpecificId()
                                    : SemIR::SpecificId::None);
  ReplaceInstBeforeConstantUse(context, decl_id, function_decl);
  return {decl_id, function_decl.function_id};
}

auto StartFunctionDefinition(Context& context, SemIR::InstId decl_id,
                             SemIR::FunctionId function_id) -> void {
  // Create the function scope and the entry block.
  context.scope_stack().PushForFunctionBody(decl_id);
  context.inst_block_stack().Push();
  context.region_stack().PushRegion(context.inst_block_stack().PeekOrAdd());
  StartGenericDefinition(context,
                         context.functions().Get(function_id).generic_id);

  CheckFunctionDefinitionSignature(context, function_id);
}

auto FinishFunctionDefinition(Context& context, SemIR::FunctionId function_id)
    -> void {
  context.inst_block_stack().Pop();
  context.scope_stack().Pop(/*check_unused=*/true);

  auto& function = context.functions().Get(function_id);
  function.body_block_ids = context.region_stack().PopRegion();

  // If this is a generic function, collect information about the definition.
  FinishGenericDefinition(context, function.generic_id);
}

}  // namespace Carbon::Check
