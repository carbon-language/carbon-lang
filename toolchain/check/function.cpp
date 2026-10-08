// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/function.h"

#include "common/find.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/action.h"
#include "toolchain/check/control_flow.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/merge.h"
#include "toolchain/check/pattern.h"
#include "toolchain/check/pattern_match.h"
#include "toolchain/check/return.h"
#include "toolchain/check/scope_stack.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/pattern.h"

namespace Carbon::Check {

auto FindSelfPattern(Context& context,
                     SemIR::InstBlockId implicit_param_patterns_id,
                     SemIR::InstBlockId param_patterns_id) -> SemIR::InstId {
  auto is_self_pattern = [&](auto param_id) {
    return SemIR::IsSelfPattern(context.sem_ir(), param_id);
  };
  // `self` is the first explicit parameter. We also look in the implicit
  // parameter list for error recovery: declaring `self` there is diagnosed (see
  // `SelfInImplicitParamList`), but we still treat it as `self` afterwards.
  auto param_patterns = context.inst_blocks().GetOrEmpty(param_patterns_id);
  if (auto self_id = FindIfOrNone(param_patterns, is_self_pattern);
      self_id.has_value()) {
    return self_id;
  }
  auto implicit_param_patterns =
      context.inst_blocks().GetOrEmpty(implicit_param_patterns_id);
  return FindIfOrNone(implicit_param_patterns, is_self_pattern);
}

auto AddReturnPattern(Context& context, SemIR::LocId loc_id,
                      Context::FormExpr form_expr) -> SemIR::InstId {
  auto result_type_id = GetPatternType(context, form_expr.type_component_id);
  auto result_type_inst_id = context.types().GetTypeInstId(result_type_id);
  auto result_id = HandleAction<SemIR::OutFormParamPatternAction>(
      context, loc_id, result_type_inst_id,
      {.type_id = SemIR::InstType::TypeId, .form_id = form_expr.form_inst_id});
  return AddInst<SemIR::ReturnSlotPattern>(
      context, loc_id,
      {.type_id = result_type_id,
       .subpattern_id = result_id,
       .type_inst_id = form_expr.type_component_inst_id});
}

auto PopReturnDecl(Context& context) -> ReturnDeclInfo {
  auto [node_id, pattern_id] =
      context.node_stack().PopWithNodeIdIf<Parse::NodeCategory::ReturnDecl>();
  if (!pattern_id) {
    return {};
  }
  if (*pattern_id == SemIR::AutoType::TypeInstId) {
    // For a deduced return type, `HandleReturnDecl` pushes `auto` in place of
    // a return pattern, and doesn't push a return form.
    return {.node_id = node_id, .is_deduced = true};
  }
  return {.node_id = node_id,
          .pattern_id = *pattern_id,
          .form = context.PopReturnForm()};
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
  SemIR::InstId return_pattern_id = SemIR::InstId::None;
  SemIR::InstId self_param_id = SemIR::InstId::None;
};
}  // namespace

// Handles construction of the signature's parameter and return types.
static auto MakeFunctionSignature(Context& context, SemIR::LocId loc_id,
                                  const FunctionDeclArgs& args)
    -> FunctionSignatureInsts {
  FunctionSignatureInsts insts;

  StartFunctionSignature(context);

  // Build and add the explicit parameters, with a leading `self` parameter if
  // one is needed. `self` is the first explicit parameter (proposal #7016),
  // matching the convention used by user-written and prelude signatures.
  // Keeping the placement consistent matters in particular for the
  // `Destroy.Op` / `Copy.Op` functions backing custom witnesses: a mismatch
  // would force a signature-adapting thunk for every witness.
  context.full_pattern_stack().StartExplicitParamList();
  if (!args.self_type_id.has_value() && args.param_type_ids.empty()) {
    insts.param_patterns_id = SemIR::InstBlockId::Empty;
  } else {
    llvm::SmallVector<SemIR::InstId> param_patterns;
    if (args.self_type_id.has_value()) {
      auto self_type_region_id = MakeEmptyRegion(
          context, context.types().GetTypeInstId(args.self_type_id));
      insts.self_param_id = AddParamPattern(
          context, loc_id, SemIR::NameId::SelfValue, self_type_region_id,
          args.self_type_id, args.self_kind);
      param_patterns.push_back(insts.self_param_id);
    }
    for (auto [param_type_id, param_kind] :
         llvm::zip_equal(args.param_type_ids, args.param_kinds)) {
      auto param_type_region_id = MakeEmptyRegion(
          context, context.types().GetTypeInstId(param_type_id));
      param_patterns.push_back(
          AddParamPattern(context, loc_id, SemIR::NameId::Underscore,
                          param_type_region_id, param_type_id, param_kind));
    }
    insts.param_patterns_id = context.inst_blocks().Add(param_patterns);
  }
  context.full_pattern_stack().EndExplicitParamList();

  if (args.return_form.form_inst_id.has_value()) {
    insts.return_type_inst_id = args.return_form.type_component_inst_id;
    insts.return_form_inst_id = args.return_form.form_inst_id;
    insts.return_pattern_id =
        AddReturnPattern(context, loc_id, args.return_form);
  }

  auto match_results =
      CalleePatternMatch(context, insts.implicit_param_patterns_id,
                         insts.param_patterns_id, insts.return_pattern_id);
  insts.call_param_patterns_id = match_results.call_param_patterns_id;
  insts.call_params_id = match_results.call_params_id;
  insts.call_param_patterns_id = match_results.call_param_patterns_id;
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
              .return_pattern_id = insts.return_pattern_id,
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
    if (new_function.name_id == SemIR::NameId::CppOperator &&
        !prev_return_type_id.has_value()) {
      return true;
    }
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
                              bool check_syntax, bool diagnose) -> bool {
  if (!CheckRedeclParamsMatch(context, DeclParams(new_function),
                              DeclParams(prev_function), prev_specific_id,
                              diagnose, check_syntax)) {
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

// Checks that the return type of a function definition is suitable for a
// definition. `return_call_param` is the `Call` parameter for the return, if
// any.
static auto CheckDefinitionReturnType(Context& context,
                                      SemIR::InstId return_pattern_id,
                                      SemIR::InstId return_call_param) -> void {
  CheckFunctionReturnPatternType(context, SemIR::LocId(return_pattern_id),
                                 return_pattern_id, SemIR::SpecificId::None);

  // `CheckFunctionReturnPatternType` should have diagnosed incomplete types,
  // so don't `RequireCompleteType` on the return type.
  if (return_call_param.has_value()) {
    // TODO: If the types are already checked for completeness then this does
    // nothing?
    TryToCompleteType(context, context.insts().Get(return_call_param).type_id(),
                      SemIR::LocId(return_call_param));
  }
}

auto CheckFunctionDefinitionSignature(Context& context,
                                      SemIR::FunctionId function_id) -> void {
  auto& function = context.functions().Get(function_id);

  auto params_to_complete =
      context.inst_blocks().GetOrEmpty(function.call_params_id);

  // The return parameter will be diagnosed after and differently from other
  // parameters.
  auto return_call_param = SemIR::InstId::None;
  if (!params_to_complete.empty() && function.return_pattern_id.has_value()) {
    return_call_param = params_to_complete.consume_back();
  }

  // Check the parameter types are complete.
  for (auto param_ref_id : params_to_complete) {
    if (param_ref_id == SemIR::ErrorInst::InstId) {
      continue;
    }

    // The parameter types need to be complete.
    RequireCompleteType(
        context, context.insts().Get(param_ref_id).type_id(),
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
  if (function.return_pattern_id.has_value()) {
    CheckDefinitionReturnType(context, function.return_pattern_id,
                              return_call_param);
  }
}

auto SetDeducedReturnType(Context& context, SemIR::FunctionId function_id,
                          SemIR::LocId loc_id, SemIR::TypeId type_id) -> void {
  auto& function = context.functions().Get(function_id);
  CARBON_CHECK(function.has_undeduced_return_type());

  // A function with a deduced return type can't be redeclared, so there is only
  // one declaration to complete.
  auto decl_id = function.latest_decl_id();
  auto decl = context.insts().GetAs<SemIR::FunctionDecl>(decl_id);

  // Reopen the function's declaration and pattern blocks to finish building the
  // signature. Add the return form and return pattern, as `HandleReturnDecl`
  // would for an explicitly declared return type, followed by the callee
  // pattern-match IR for the return, as `PopNameComponent` would.
  context.inst_block_stack().Push(
      SemIR::InstBlockId::None,
      context.inst_blocks().GetOrEmpty(decl.decl_block_id));
  context.pattern_block_stack().Push(
      SemIR::InstBlockId::None,
      context.inst_blocks().GetOrEmpty(function.pattern_block_id));
  auto form = Context::FormExpr::Error;
  if (type_id != SemIR::ErrorInst::TypeId) {
    // Represent the return type as a type literal at the location it was
    // deduced from. This is a new instruction, rather than the canonical
    // instruction for the type, so that in a generic it is attached to the
    // current region, and can depend on anything declared before this point.
    auto type_inst_id = AddTypeInst<SemIR::TypeLiteral>(
        context, loc_id,
        {.type_id = SemIR::TypeType::TypeId,
         .value_id = context.types().GetTypeInstId(type_id)});
    auto form_inst_id = AddInst(
        context, SemIR::LocIdAndInst::RuntimeVerified(
                     context.sem_ir(), loc_id,
                     SemIR::InitForm{.type_id = SemIR::FormType::TypeId,
                                     .type_component_inst_id = type_inst_id}));
    form = {.form_inst_id = form_inst_id,
            .type_component_inst_id = type_inst_id,
            .type_component_id =
                context.types().GetTypeIdForTypeInstId(type_inst_id)};
  }
  auto return_pattern_id = AddReturnPattern(context, loc_id, form);
  auto results = CalleeReturnPatternMatch(
      context,
      {.call_param_patterns_id = function.call_param_patterns_id,
       .call_params_id = function.call_params_id,
       .param_ranges = function.call_param_ranges},
      return_pattern_id);
  function.pattern_block_id = context.pattern_block_stack().Pop();
  decl.decl_block_id = context.inst_block_stack().Pop();
  ReplaceInstPreservingConstantValue(context, decl_id, decl);

  function.call_param_patterns_id = results.call_param_patterns_id;
  function.call_params_id = results.call_params_id;
  function.call_param_ranges = results.param_ranges;
  function.return_type_inst_id = form.type_component_inst_id;
  function.return_form_inst_id = form.form_inst_id;
  function.return_pattern_id = return_pattern_id;
  CARBON_CHECK(!function.has_undeduced_return_type());

  if (type_id != SemIR::ErrorInst::TypeId) {
    auto return_call_param = SemIR::InstId::None;
    if (results.param_ranges.return_size() == 1) {
      return_call_param = context.inst_blocks().Get(
          results.call_params_id)[results.param_ranges.return_begin().index];
    }
    CheckDefinitionReturnType(context, return_pattern_id, return_call_param);
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

// Diagnoses when positional params aren't supported. Reassigns the pattern
// block if needed.
static auto DiagnosePositionalParams(Context& context,
                                     SemIR::Function& function_info) -> void {
  if (function_info.param_patterns_id.has_value()) {
    return;
  }

  context.TODO(function_info.latest_decl_id(),
               "function with positional parameters");
  function_info.param_patterns_id = SemIR::InstBlockId::Empty;
}

// For the top-level parameter patterns list, and for any level of nested tuple
// patterns, ensure that if a subpattern provides a default value, all
// subsequent patterns at that level of nesting must provide a default value as
// well. Returns the number of default values provided at the top level of the
// function parameter, useful for efficient arity checking in callers later on.
//
// TODO: per https://github.com/carbon-language/carbon-lang/issues/7529, this
// should also consider automatically supplied defaults for fully-specified
// tuple subpatterns, and consider them as having a default for the purposes
// of the out-of-order detection. It will also need to detect the error
// condition when a default is also specified for those fully-specified tuple
// subpatterns.
static auto CheckDefaults(Context& context, SemIR::Function& function)
    -> int32_t {
  if (!function.param_patterns_id.has_value()) {
    return 0;
  }

  struct PatternLevelState {
    // The inst ids of the subpatterns on this level of tuple subpattern
    // nesting, treated as a work list, so in reverse order of declaration.
    llvm::SmallVector<SemIR::InstId> subpattern_ids;

    // If patterns at this level of nesting have default values, this refers
    // to the first instruction to specify a default, useful for diagnostics.
    SemIR::InstId first_pattern_with_default = SemIR::InstId::None;

    // If we encounter a tuple-pattern during processing, we suspend processing
    // of this pattern level, in the middle of processing a single pattern from
    // root to leaves. So we record the current state of processing of a single
    // pattern to return to it after processing any tuple subpatterns.

    // True if the current pattern being processed has a default value
    // specified.
    bool current_pattern_has_default = false;

    // The current pattern we are processing, stored separately since it's been
    // popped from the `pattern_work_list` and already processed, just may need
    // subsequent processing.
    SemIR::InstId current_id = SemIR::InstId::None;

    // A work list of patterns to be processed at this level of nesting.
    llvm::SmallVector<SemIR::InstId> pattern_work_list;

    // A list of subpatterns missing required defaults, to coalesce error
    // reporting into a single diagnostic.
    llvm::SmallVector<SemIR::InstId> patterns_missing_defaults;

    // A count of the number of patterns on this level that have defaults.
    int32_t default_count = 0;
  };

  llvm::SmallVector<PatternLevelState> level_state_stack;
  size_t default_count = 0;
  level_state_stack.push_back({});
  llvm::append_range(
      level_state_stack.back().subpattern_ids,
      llvm::reverse(context.inst_blocks().Get(function.param_patterns_id)));

  while (!level_state_stack.empty()) {
    PatternLevelState* state = &level_state_stack.back();
    while (!state->subpattern_ids.empty() ||
           !state->pattern_work_list.empty() || state->current_id.has_value()) {
      // If we're not resuming processing a pattern from a nested state, start
      // processing the next subpattern.
      if (!state->current_id.has_value()) {
        state->pattern_work_list.push_back(
            state->subpattern_ids.pop_back_val());
        state->current_pattern_has_default = false;
      }
      while (!state->pattern_work_list.empty()) {
        state->current_id = state->pattern_work_list.pop_back_val();
        auto inst = context.insts().Get(state->current_id);
        CARBON_KIND_SWITCH(inst) {
          case CARBON_KIND(SemIR::DefaultValuePattern default_value_pattern): {
            state->current_pattern_has_default = true;
            state->default_count += 1;
            state->pattern_work_list.push_back(
                default_value_pattern.subpattern_id);
            break;
          }
          case CARBON_KIND(
              SemIR::WrapperBindingPattern wrapper_binding_pattern): {
            state->pattern_work_list.push_back(
                wrapper_binding_pattern.subpattern_id);
            break;
          }
          case CARBON_KIND(SemIR::TuplePattern tuple_pattern): {
            auto elements =
                context.inst_blocks().Get(tuple_pattern.elements_id);
            if (!elements.empty()) {
              // Start a new state for the nested tuple pattern elements.
              level_state_stack.push_back({});
              state = &level_state_stack.back();
              llvm::append_range(state->subpattern_ids,
                                 llvm::reverse(elements));
            }
            break;
          }
          default:
            // We only process patterns containing subpatterns, so this is an
            // intentional no-op.
            break;
        }
      }
      // Finished processing this subpattern, detect a missing default if
      // required.
      if (state->current_pattern_has_default &&
          !state->first_pattern_with_default.has_value()) {
        state->first_pattern_with_default = state->current_id;
      } else if (!state->current_pattern_has_default &&
                 state->first_pattern_with_default.has_value()) {
        state->patterns_missing_defaults.push_back(state->current_id);
      }
      state->current_id = SemIR::InstId::None;
    }
    // Finished processing this tuple-pattern, emit diagnostics if any.
    if (!state->patterns_missing_defaults.empty()) {
      CARBON_DIAGNOSTIC(RequiredPatternDefaultValueMissing, Error,
                        "this pattern is missing a required default value.");
      CARBON_DIAGNOSTIC(RequiredPatternDefaultValueFirstDefault, Note,
                        "all patterns to the right of this first pattern with "
                        "a default value must also specify a default value.");
      CARBON_DIAGNOSTIC(
          RequiredPatternDefaultValueMissingAdditional, Note,
          "this pattern is also missing a required default value.");
      auto inst_ref = llvm::ArrayRef(state->patterns_missing_defaults);
      auto builder = context.emitter().Build(
          inst_ref.consume_front(), RequiredPatternDefaultValueMissing);
      for (auto inst_id : inst_ref) {
        builder.Note(inst_id, RequiredPatternDefaultValueMissingAdditional);
      }
      builder.Note(state->first_pattern_with_default,
                   RequiredPatternDefaultValueFirstDefault);
      builder.Emit();
    }

    // Extract the count from the level we just completed, overwriting any
    // nested level value extracted previously.
    default_count = level_state_stack.back().default_count;
    level_state_stack.pop_back();
  }

  return default_count;
}

auto CheckFunctionParams(Context& context, SemIR::Function& function) -> void {
  function.default_value_arity = CheckDefaults(context, function);
  DiagnosePositionalParams(context, function);
}

auto StartFunctionDefinition(Context& context, SemIR::InstId decl_id,
                             SemIR::FunctionId function_id) -> void {
  // Create the function scope and the entry block.
  context.scope_stack().PushForFunctionBody(decl_id);
  context.inst_block_stack().Push();
  context.observe_stack().PushArray();
  context.region_stack().PushRegion(context.inst_block_stack().PeekOrAdd());
  StartGenericDefinition(context,
                         context.functions().Get(function_id).generic_id);

  CheckFunctionDefinitionSignature(context, function_id);
}

auto CheckFunctionReturnOnFinish(Context& context, Parse::NodeId node_id,
                                 SemIR::FunctionId function_id) -> void {
  bool is_end_reachable = IsCurrentPositionReachable(context);

  // If the return type is deduced and we've not seen a `returned var`, the
  // return type is determined by the `return` statements in the body.
  if (context.functions().Get(function_id).has_undeduced_return_type() &&
      !DeduceReturnTypeAtEndOfBody(context, function_id)) {
    // There are no `return` statements, which has already been diagnosed.
    return;
  }

  // If the `}` of the function is reachable, reject if we need a return value
  // and otherwise add an implicit `return;`.
  if (is_end_reachable) {
    if (context.functions().Get(function_id).return_form_inst_id.has_value()) {
      CARBON_DIAGNOSTIC(
          MissingReturnStatement, Error,
          "missing `return` at end of function with declared return type");
      context.emitter().Emit(LocIdForDiagnostics::TokenOnly(node_id),
                             MissingReturnStatement);
    } else {
      AddReturnInstWithCleanups(context, node_id);
    }
  }
}

auto FinishFunctionDefinition(Context& context, SemIR::FunctionId function_id)
    -> void {
  context.inst_block_stack().Pop();
  // Any cleanups for a function will have been handled when emitting `return`s.
  context.scope_stack().DiscardCleanupsSince(
      context.scope_stack().function_cleanup_scope_depth());
  context.scope_stack().Pop(/*check_unused=*/true);

  auto observe_block_id =
      context.observe_blocks().Add(context.observe_stack().PeekArray());
  context.observe_stack().PopArray();

  auto& function = context.functions().Get(function_id);
  function.body_block_ids = context.region_stack().PopRegion();
  function.observe_block_id = observe_block_id;

  // If this is a generic function, collect information about the definition.
  FinishGenericDefinition(context, function.generic_id);
}

}  // namespace Carbon::Check
