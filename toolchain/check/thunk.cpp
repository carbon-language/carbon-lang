// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/thunk.h"

#include <utility>

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/call.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/cpp/operators.h"
#include "toolchain/check/deferred_definition_worklist.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/function.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/member_access.h"
#include "toolchain/check/name_ref.h"
#include "toolchain/check/pattern.h"
#include "toolchain/check/pattern_match.h"
#include "toolchain/check/pointer_dereference.h"
#include "toolchain/check/return.h"
#include "toolchain/check/type.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/pattern.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Adds a pattern instruction for a thunk, copying the location from an existing
// instruction.
static auto RebuildPatternInst(Context& context, SemIR::InstId orig_inst_id,
                               SemIR::Inst new_inst) -> SemIR::InstId {
  // Ensure we built the same kind of instruction. In particular, this ensures
  // that the location of the old instruction can be reused for the new one.
  CARBON_CHECK(context.insts().Get(orig_inst_id).kind() == new_inst.kind(),
               "Rebuilt pattern with the wrong kind: {0} -> {1}",
               context.insts().Get(orig_inst_id), new_inst);
  return AddPatternInst(context, SemIR::LocIdAndInst::UncheckedLoc(
                                     SemIR::LocId(orig_inst_id), new_inst));
}

// Wrapper to allow the type to be specified as a template argument for API
// consistency with `AddInst`.
template <typename InstT>
static auto RebuildPatternInst(Context& context, SemIR::InstId orig_inst_id,
                               InstT new_inst) -> SemIR::InstId {
  return RebuildPatternInst(context, orig_inst_id, SemIR::Inst(new_inst));
}

// Makes a copy of the given binding pattern, with its type adjusted to be
// `new_pattern_type_id`.
static auto CloneBindingPattern(Context& context, SemIR::InstId pattern_id,
                                SemIR::AnyBindingPattern pattern,
                                SemIR::TypeId new_pattern_type_id)
    -> SemIR::InstId {
  auto entity_name = context.entity_names().Get(pattern.entity_name_id);
  CARBON_CHECK((pattern.kind == SemIR::SymbolicBindingPattern::Kind) ==
               entity_name.bind_index().has_value());
  CARBON_CHECK((pattern.kind == SemIR::FormBindingPattern::Kind) ==
               entity_name.form_id.has_value());
  if (pattern.kind == SemIR::FormBindingPattern::Kind) {
    context.TODO(pattern_id, "Support for cloning form bindings");
    return SemIR::ErrorInst::InstId;
  }
  // Get the transformed type of the binding.
  if (new_pattern_type_id == SemIR::ErrorInst::TypeId) {
    return SemIR::ErrorInst::InstId;
  }
  auto type_inst_id = context.types()
                          .GetAs<SemIR::PatternType>(new_pattern_type_id)
                          .scrutinee_type_inst_id;
  auto type_id = context.types().GetTypeIdForTypeInstId(type_inst_id);
  auto type_expr_region_id = context.sem_ir().expr_regions().Add(
      {.block_ids = {SemIR::InstBlockId::Empty}, .result_id = type_inst_id});

  // Rebuild the binding pattern.
  return AddBindingPattern(
             context, SemIR::LocId(pattern_id), entity_name.name_id, type_id,
             /*form_id=*/SemIR::ConstantId::None, type_expr_region_id,
             pattern.kind, entity_name.is_template,
             /*is_unused=*/false)
      .pattern_id;
}

// Makes a copy of the given pattern instruction, substituting values from a
// specific as needed. The resulting pattern behaves like a newly-created
// pattern, so is suitable for running `CalleePatternMatch` against.
static auto ClonePattern(Context& context, SemIR::SpecificId specific_id,
                         SemIR::InstId pattern_id) -> SemIR::InstId {
  if (!pattern_id.has_value()) {
    return SemIR::InstId::None;
  }

  auto get_type = [&](SemIR::InstId inst_id) -> SemIR::TypeId {
    return SemIR::GetTypeOfInstInSpecific(context.sem_ir(), specific_id,
                                          inst_id);
  };

  auto pattern = context.insts().Get(pattern_id);

  // Decompose the pattern. The forms we allow for patterns in a function
  // parameter list are currently fairly restrictive.

  // Optional parameter pattern.
  auto [param, param_id] = context.insts().TryUnwrap(
      pattern, pattern_id, &SemIR::AnyParamPattern::subpattern_id);

  // Finally, either a binding pattern or a return slot pattern.
  auto new_pattern_id = SemIR::InstId::None;
  if (auto binding = pattern.TryAs<SemIR::AnyBindingPattern>()) {
    new_pattern_id = CloneBindingPattern(context, pattern_id, *binding,
                                         get_type(pattern_id));
  } else if (auto return_slot = pattern.TryAs<SemIR::ReturnSlotPattern>()) {
    new_pattern_id = RebuildPatternInst<SemIR::ReturnSlotPattern>(
        context, pattern_id,
        {.type_id = get_type(pattern_id),
         .type_inst_id = SemIR::TypeInstId::None});
  } else {
    CARBON_CHECK(pattern.Is<SemIR::ErrorInst>(),
                 "Unexpected pattern {0} in function signature", pattern);
    return SemIR::ErrorInst::InstId;
  }

  // Rebuild parameter.
  if (param) {
    new_pattern_id = RebuildPatternInst<SemIR::AnyParamPattern>(
        context, param_id,
        {.kind = param->kind,
         .type_id = get_type(param_id),
         .subpattern_id = new_pattern_id});
  }

  return new_pattern_id;
}

static auto ClonePatternBlock(Context& context, SemIR::SpecificId specific_id,
                              SemIR::InstBlockId inst_block_id)
    -> SemIR::InstBlockId {
  if (!inst_block_id.has_value()) {
    return SemIR::InstBlockId::None;
  }
  return context.inst_blocks().Transform(
      inst_block_id, [&](SemIR::InstId inst_id) {
        return ClonePattern(context, specific_id, inst_id);
      });
}

static auto CloneInstId(Context& context, SemIR::SpecificId specific_id,
                        SemIR::InstId inst_id) -> SemIR::InstId {
  if (!inst_id.has_value()) {
    return SemIR::InstId::None;
  }

  return GetOrAddInst<SemIR::SpecificConstant>(
      context, SemIR::LocId(inst_id),
      {.type_id = SemIR::TypeType::TypeId,
       .inst_id = inst_id,
       .specific_id = specific_id});
}

static auto CloneTypeInstId(Context& context, SemIR::SpecificId specific_id,
                            SemIR::TypeInstId inst_id) -> SemIR::TypeInstId {
  if (!inst_id.has_value()) {
    return SemIR::TypeInstId::None;
  }

  return context.types().GetAsTypeInstId(
      CloneInstId(context, specific_id, inst_id));
}

static auto CloneFunctionDecl(Context& context, SemIR::LocId loc_id,
                              SemIR::FunctionId signature_id,
                              SemIR::SpecificId signature_specific_id,
                              SemIR::FunctionId callee_id)
    -> std::pair<SemIR::FunctionId, SemIR::InstId> {
  StartGenericDecl(context);

  const auto& signature = context.functions().Get(signature_id);

  // Clone the signature.
  context.pattern_block_stack().Push();
  auto implicit_param_patterns_id = ClonePatternBlock(
      context, signature_specific_id, signature.implicit_param_patterns_id);
  auto param_patterns_id = ClonePatternBlock(context, signature_specific_id,
                                             signature.param_patterns_id);
  auto return_patterns_id = ClonePatternBlock(context, signature_specific_id,
                                              signature.return_patterns_id);
  auto return_type_inst_id = CloneTypeInstId(context, signature_specific_id,
                                             signature.return_type_inst_id);
  auto return_form_inst_id = CloneInstId(context, signature_specific_id,
                                         signature.return_form_inst_id);
  auto self_param_id = FindSelfPattern(context, implicit_param_patterns_id);
  auto pattern_block_id = context.pattern_block_stack().Pop();

  // Perform callee-side pattern matching to rebuild the parameter list.
  context.inst_block_stack().Push();
  auto match_results =
      CalleePatternMatch(context, implicit_param_patterns_id, param_patterns_id,
                         return_patterns_id);
  auto decl_block_id = context.inst_block_stack().Pop();

  // Create the `FunctionDecl` instruction.
  auto& callee = context.functions().Get(callee_id);
  auto [decl_id, function_id] = MakeFunctionDecl(
      context, loc_id, decl_block_id, /*build_generic=*/true,
      /*is_definition=*/true,
      SemIR::Function{
          {
              .name_id = signature.name_id,
              .parent_scope_id = callee.parent_scope_id,
              // Set by `MakeFunctionDecl`.
              .generic_id = SemIR::GenericId::None,
              .first_param_node_id = signature.first_param_node_id,
              .last_param_node_id = signature.last_param_node_id,
              .pattern_block_id = pattern_block_id,
              .implicit_param_patterns_id = implicit_param_patterns_id,
              .param_patterns_id = param_patterns_id,
              .is_extern = false,
              .extern_library_id = SemIR::LibraryNameId::None,
              .non_owning_decl_id = SemIR::InstId::None,
              // Set by `MakeFunctionDecl`.
              .first_owning_decl_id = SemIR::InstId::None,
          },
          {
              .call_param_patterns_id = match_results.call_param_patterns_id,
              .call_params_id = match_results.call_params_id,
              .call_param_ranges = match_results.param_ranges,
              .return_type_inst_id = return_type_inst_id,
              .return_form_inst_id = return_form_inst_id,
              .return_patterns_id = return_patterns_id,
              .virtual_modifier = callee.virtual_modifier,
              .virtual_index = callee.virtual_index,
              .evaluation_mode = signature.evaluation_mode,
              .self_param_id = self_param_id,
          }});
  context.inst_block_stack().AddInstId(decl_id);
  return {function_id, decl_id};
}

static auto HasDeclaredReturnType(Context& context,
                                  SemIR::FunctionId function_id) -> bool {
  return context.functions().Get(function_id).return_type_inst_id.has_value();
}

auto PerformThunkCall(Context& context, SemIR::LocId loc_id,
                      SemIR::FunctionId function_id,
                      llvm::ArrayRef<SemIR::InstId> call_arg_ids,
                      SemIR::InstId callee_id) -> SemIR::InstId {
  auto& function = context.functions().Get(function_id);

  auto param_pattern_ids =
      context.inst_blocks().Get(function.call_param_patterns_id);

  // Maps each `Call` parameter pattern ID to its index.
  // TODO: is it possible to arrange for the param patterns to be created in
  // order, so that we could use `param_pattern_ids` for this directly?
  struct InstWithIndex {
    SemIR::InstId inst_id;
    int index;

    auto operator<(InstWithIndex other) const -> bool {
      return inst_id.index < other.inst_id.index;
    }
  };
  llvm::SmallVector<InstWithIndex> param_to_index;

  param_to_index.reserve(param_pattern_ids.size());
  for (auto [index, inst_id] : llvm::enumerate(param_pattern_ids)) {
    param_to_index.push_back({inst_id, static_cast<int>(index)});
  }
  llvm::sort(param_to_index);

  // Given that `call_arg_ids` is a list of the _`Call`_ arguments for a call to
  // `function_id`, this returns the _syntactic_ argument that was passed for
  // param_pattern_id in that call.
  auto build_syntactic_arg = [&](SemIR::InstId param_pattern_id) {
    // NOLINTNEXTLINE(readability-qualified-auto)
    auto result =
        llvm::lower_bound(param_to_index, InstWithIndex{param_pattern_id, -1});
    if (result < param_to_index.end() && result->inst_id == param_pattern_id) {
      return call_arg_ids[result->index];
    } else {
      if (param_pattern_id != SemIR::ErrorInst::InstId) {
        context.TODO(param_pattern_id,
                     "don't know how to reconstruct the syntactic argument for "
                     "this pattern in thunk");
      }
      return SemIR::ErrorInst::InstId;
    }
  };

  llvm::SmallVector<SemIR::InstId> args;

  // If we have a self parameter, form `self.<callee_id>`.
  if (function.self_param_id.has_value()) {
    auto self_arg_id = build_syntactic_arg(function.self_param_id);
    if (IsCppConstructorOrNonMethodOperator(context, callee_id)) {
      // When calling a C++ constructor to implement `Copy`, or calling a C++
      // non-method operator to implement a Carbon operator, the interface has a
      // `self` parameter but C++ models that parameter as an explicit argument
      // instead, so add the `self` to the argument list instead in that case.
      args.push_back(self_arg_id);
    } else {
      callee_id =
          PerformCompoundMemberAccess(context, loc_id, self_arg_id, callee_id);
    }
  }

  // Form an argument list.
  for (auto pattern_id :
       context.inst_blocks().Get(function.param_patterns_id)) {
    args.push_back(build_syntactic_arg(pattern_id));
  }

  return PerformCall(context, loc_id, callee_id, args);
}

// Build a call to a function that forwards the arguments of the enclosing
// function, for use when constructing a thunk.
static auto BuildThunkCall(Context& context, SemIR::FunctionId function_id,
                           SemIR::InstId callee_id) -> SemIR::InstId {
  auto& function = context.functions().Get(function_id);

  // Build a `NameRef` naming the callee, and a `SpecificConstant` if needed.
  auto loc_id = SemIR::LocId(callee_id);
  auto callee_type = context.types().GetAs<SemIR::FunctionType>(
      context.insts().Get(callee_id).type_id());
  callee_id = BuildNameRef(context, loc_id, function.name_id, callee_id,
                           callee_type.specific_id);

  // Build a reference to each parameter for use as call arguments.
  llvm::SmallVector<SemIR::InstId> call_args;
  auto call_params = context.inst_blocks().Get(function.call_params_id);
  call_args.reserve(call_params.size());
  for (auto call_param_id : call_params) {
    // Use a pretty name for the `name_ref`. While it's suspicious to use a
    // pretty name in the IR like this, the only reason we include a name at all
    // here is to make the formatted SemIR more readable.
    auto call_param = context.insts().GetAs<SemIR::AnyParam>(call_param_id);
    call_args.push_back(BuildNameRef(context, SemIR::LocId(call_param_id),
                                     call_param.pretty_name_id, call_param_id,
                                     SemIR::SpecificId::None));
  }

  return PerformThunkCall(context, loc_id, function_id, call_args, callee_id);
}

// Given a declaration of a thunk and the function that it should call, build
// the thunk body.
static auto BuildThunkDefinition(Context& context,
                                 SemIR::FunctionId signature_id,
                                 SemIR::FunctionId function_id,
                                 SemIR::InstId thunk_id,
                                 SemIR::InstId callee_id) {
  // TODO: Improve the diagnostics produced here. Specifically, it would likely
  // be better for the primary error message to be that we tried to produce a
  // thunk because of a type mismatch, but couldn't, with notes explaining
  // why, rather than the primary error message being whatever went wrong
  // building the thunk.

  {
    // The check below produces diagnostics referring to the signature, so also
    // note the callee.
    Diagnostics::AnnotationScope annot_scope(
        &context.emitter(), [&](DiagnosticBuilder& builder) {
          CARBON_DIAGNOSTIC(ThunkCallee, Note,
                            "while building thunk calling this function");
          builder.Note(callee_id, ThunkCallee);
        });

    StartFunctionDefinition(context, thunk_id, function_id);
  }

  // The checks below produce diagnostics pointing at the callee, so also note
  // the signature.
  Diagnostics::AnnotationScope annot_scope(
      &context.emitter(), [&](DiagnosticBuilder& builder) {
        CARBON_DIAGNOSTIC(
            ThunkSignature, Note,
            "while building thunk to match the signature of this function");
        builder.Note(context.functions().Get(signature_id).first_owning_decl_id,
                     ThunkSignature);
      });

  auto call_id = BuildThunkCall(context, function_id, callee_id);
  if (HasDeclaredReturnType(context, function_id)) {
    BuildReturnWithExpr(context, SemIR::LocId(callee_id), call_id);
  } else {
    DiscardExpr(context, call_id);
    BuildReturnWithNoExpr(context, SemIR::LocId(callee_id));
  }

  FinishFunctionDefinition(context, function_id);
}

auto BuildThunkDefinition(Context& context,
                          DeferredDefinitionWorklist::DefineThunk&& task)
    -> void {
  context.scope_stack().Restore(std::move(task.scope));

  BuildThunkDefinition(context, task.info.signature_id, task.info.function_id,
                       task.info.decl_id, task.info.callee_id);

  context.scope_stack().Pop();
}

auto BuildThunk(Context& context, SemIR::FunctionId signature_id,
                SemIR::SpecificId signature_specific_id,
                SemIR::InstId callee_id, bool defer_definition)
    -> SemIR::InstId {
  auto callee = SemIR::GetCalleeAsFunction(context.sem_ir(), callee_id);

  // Check whether we can use the given function without a thunk.
  // TODO: For virtual functions, we want different rules for checking `self`.
  // TODO: This is too strict; for example, we should not compare parameter
  // names here.
  if (CheckFunctionTypeMatches(
          context, context.functions().Get(callee.function_id),
          context.functions().Get(signature_id), signature_specific_id,
          /*check_syntax=*/false, /*check_self=*/true, /*diagnose=*/false)) {
    return callee_id;
  }

  // From P3763:
  //   If the function in the interface does not have a return type, the
  //   program is invalid if the function in the impl specifies a return type.
  //
  // Call into the redeclaration checking logic to produce a suitable error.
  //
  // TODO: Consider a different rule: always use an explicit return type for the
  // thunk, and always convert the result of the wrapped call to the return type
  // of the thunk.
  if (!HasDeclaredReturnType(context, signature_id) &&
      HasDeclaredReturnType(context, callee.function_id)) {
    bool success = CheckFunctionReturnTypeMatches(
        context, context.functions().Get(callee.function_id),
        context.functions().Get(signature_id), signature_specific_id);
    CARBON_CHECK(!success, "Return type unexpectedly matches");
    return SemIR::ErrorInst::InstId;
  }

  // Create a scope for the function's parameters and generic parameters.
  context.scope_stack().PushForDeclName();

  // We can't use the function directly. Build a thunk.
  // TODO: Check for and diagnose obvious reasons why this will fail, such as
  // arity mismatch, before trying to build the thunk.
  auto [function_id, thunk_id] =
      CloneFunctionDecl(context, SemIR::LocId(callee_id), signature_id,
                        signature_specific_id, callee.function_id);

  // Track that this function is a thunk.
  context.functions().Get(function_id).SetThunk(callee_id);

  if (defer_definition) {
    // Register the thunk to be defined when we reach the end of the enclosing
    // deferred definition scope, for example an `impl` or `class` definition,
    // as if the thunk's body were written inline in this location.
    context.deferred_definition_worklist().SuspendThunkAndPush(
        context, {
                     .signature_id = signature_id,
                     .function_id = function_id,
                     .decl_id = thunk_id,
                     .callee_id = callee_id,
                 });
  } else {
    BuildThunkDefinition(context, signature_id, function_id, thunk_id,
                         callee_id);
    context.scope_stack().Pop();
  }

  return thunk_id;
}

}  // namespace Carbon::Check
