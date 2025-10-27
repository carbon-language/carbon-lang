// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/thunk.h"

#include <utility>

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/call.h"
#include "toolchain/check/convert.h"
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
  return AddBindingPattern(context, SemIR::LocId(pattern_id),
                           entity_name.name_id, type_id, type_expr_region_id,
                           pattern.kind, entity_name.is_template)
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

  // Optional `addr`, only for `self`.
  auto [addr, addr_id] = context.insts().TryUnwrap(
      pattern, pattern_id, &SemIR::AddrPattern::inner_id);

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
         .subpattern_id = new_pattern_id,
         .index = SemIR::CallParamIndex::None});
  }

  // Rebuild `addr`.
  if (addr) {
    new_pattern_id = RebuildPatternInst<SemIR::AddrPattern>(
        context, addr_id,
        {.type_id = get_type(addr_id), .inner_id = new_pattern_id});
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
  auto return_slot_pattern_id = ClonePattern(context, signature_specific_id,
                                             signature.return_slot_pattern_id);
  auto self_param_id = FindSelfPattern(context, implicit_param_patterns_id);
  auto pattern_block_id = context.pattern_block_stack().Pop();

  // Perform callee-side pattern matching to rebuild the parameter list.
  context.inst_block_stack().Push();
  auto call_params_id =
      CalleePatternMatch(context, implicit_param_patterns_id, param_patterns_id,
                         return_slot_pattern_id);
  auto decl_block_id = context.inst_block_stack().Pop();

  // Create the `FunctionDecl` instruction.
  SemIR::FunctionDecl function_decl = {SemIR::TypeId::None,
                                       SemIR::FunctionId::None, decl_block_id};
  auto decl_id = AddPlaceholderInst(
      context, SemIR::LocIdAndInst::UncheckedLoc(loc_id, function_decl));
  auto generic_id = BuildGenericDecl(context, decl_id);

  // Create the `Function` object.
  auto& callee = context.functions().Get(callee_id);
  function_decl.function_id = context.functions().Add(
      SemIR::Function{{.name_id = signature.name_id,
                       .parent_scope_id = callee.parent_scope_id,
                       .generic_id = generic_id,
                       .first_param_node_id = signature.first_param_node_id,
                       .last_param_node_id = signature.last_param_node_id,
                       .pattern_block_id = pattern_block_id,
                       .implicit_param_patterns_id = implicit_param_patterns_id,
                       .param_patterns_id = param_patterns_id,
                       .is_extern = false,
                       .extern_library_id = SemIR::LibraryNameId::None,
                       .non_owning_decl_id = SemIR::InstId::None,
                       .first_owning_decl_id = decl_id,
                       .definition_id = decl_id},
                      {.call_params_id = call_params_id,
                       .return_slot_pattern_id = return_slot_pattern_id,
                       .virtual_modifier = callee.virtual_modifier,
                       .virtual_index = callee.virtual_index,
                       .self_param_id = self_param_id}});
  function_decl.type_id =
      GetFunctionType(context, function_decl.function_id,
                      context.scope_stack().PeekSpecificId());
  ReplaceInstBeforeConstantUse(context, decl_id, function_decl);
  return {function_decl.function_id, decl_id};
}

static auto HasDeclaredReturnType(Context& context,
                                  SemIR::FunctionId function_id) -> bool {
  return context.functions()
      .Get(function_id)
      .return_slot_pattern_id.has_value();
}

auto BuildThunk(Context& context, SemIR::FunctionId signature_id,
                SemIR::SpecificId signature_specific_id,
                SemIR::InstId callee_id) -> SemIR::InstId {
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

  // Register the thunk to be defined when we reach the end of the enclosing
  // deferred definition scope, for example an `impl` or `class` definition, as
  // if the thunk's body were written inline in this location.
  context.deferred_definition_worklist().SuspendThunkAndPush(
      context, {
                   .signature_id = signature_id,
                   .function_id = function_id,
                   .decl_id = thunk_id,
                   .callee_id = callee_id,
               });

  return thunk_id;
}

// Build an expression that names the value matched by a pattern.
static auto BuildPatternRef(Context& context,
                            llvm::ArrayRef<SemIR::InstId> arg_ids,
                            SemIR::InstId pattern_id) -> SemIR::InstId {
  auto pattern = context.insts().Get(pattern_id);

  auto addr = context.insts()
                  .TryUnwrap(pattern, pattern_id, &SemIR::AddrPattern::inner_id)
                  .first;

  auto pattern_ref_id = SemIR::InstId::None;
  if (auto value_param = pattern.TryAs<SemIR::ValueParamPattern>()) {
    pattern_ref_id = arg_ids[value_param->index.index];
  } else {
    if (pattern_id != SemIR::ErrorInst::InstId) {
      context.TODO(
          pattern_id,
          "don't know how to build reference to this pattern in thunk");
    }
    return SemIR::ErrorInst::InstId;
  }

  if (addr) {
    pattern_ref_id = PerformPointerDereference(
        context, SemIR::LocId(pattern_id), pattern_ref_id, [](SemIR::TypeId) {
          CARBON_FATAL("addr subpattern is not a pointer");
        });
  }

  return pattern_ref_id;
}

auto PerformThunkCall(Context& context, SemIR::LocId loc_id,
                      SemIR::FunctionId function_id,
                      llvm::ArrayRef<SemIR::InstId> call_arg_ids,
                      SemIR::InstId callee_id) -> SemIR::InstId {
  auto& function = context.functions().Get(function_id);

  // If we have a self parameter, form `self.<callee_id>`.
  if (function.self_param_id.has_value()) {
    callee_id = PerformCompoundMemberAccess(
        context, loc_id,
        BuildPatternRef(context, call_arg_ids, function.self_param_id),
        callee_id);
  }

  // Form an argument list.
  llvm::SmallVector<SemIR::InstId> args;
  for (auto pattern_id :
       context.inst_blocks().Get(function.param_patterns_id)) {
    args.push_back(BuildPatternRef(context, call_arg_ids, pattern_id));
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

    CheckFunctionDefinitionSignature(context, function_id);
  }

  // TODO: This duplicates much of the handling for FunctionDefinitionStart and
  // FunctionDefinition parse nodes. Consider refactoring.
  context.scope_stack().PushForFunctionBody(thunk_id);
  context.inst_block_stack().Push();
  context.region_stack().PushRegion(context.inst_block_stack().PeekOrAdd());
  StartGenericDefinition(context,
                         context.functions().Get(function_id).generic_id);

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

  context.inst_block_stack().Pop();
  context.scope_stack().Pop();

  auto& function = context.functions().Get(function_id);
  function.body_block_ids = context.region_stack().PopRegion();
  FinishGenericDefinition(context, function.generic_id);
}

auto BuildThunkDefinition(Context& context,
                          DeferredDefinitionWorklist::DefineThunk&& task)
    -> void {
  context.scope_stack().Restore(std::move(task.scope));

  BuildThunkDefinition(context, task.info.signature_id, task.info.function_id,
                       task.info.decl_id, task.info.callee_id);

  context.scope_stack().Pop();
}

}  // namespace Carbon::Check
