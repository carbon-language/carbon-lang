// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/thunk.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/call.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/function.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/member_access.h"
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

static auto ClonePattern(Context& context, SemIR::SpecificId specific_id,
                         SemIR::InstId pattern_id) -> SemIR::InstId {
  if (!pattern_id.has_value()) {
    return SemIR::InstId::None;
  }

  auto pattern = context.insts().Get(pattern_id);

  // Decompose the pattern. The forms we allow for patterns in a function
  // parameter list are currently fairly restrictive.

  // Optional `addr`, only for `self`.
  auto addr = pattern.TryAs<SemIR::AddrPattern>();
  auto addr_id = pattern_id;
  if (addr) {
    pattern_id = addr->inner_id;
    pattern = context.insts().Get(pattern_id);
  }

  // Optional parameter pattern.
  auto param = pattern.TryAs<SemIR::AnyParamPattern>();
  auto param_id = pattern_id;
  if (param) {
    pattern_id = param->subpattern_id;
    pattern = context.insts().Get(pattern_id);
  }

  // Finally, either a binding pattern or a return slot pattern.
  auto new_pattern_id = SemIR::InstId::None;
  auto inner_type_id =
      SemIR::GetTypeOfInstInSpecific(context.sem_ir(), specific_id, pattern_id);
  if (auto binding = pattern.TryAs<SemIR::AnyBindingPattern>()) {
    // TODO: This duplicates some of the work done by `HandleAnyBindingPattern`.
    bool is_generic = pattern.Is<SemIR::SymbolicBindingPattern>();

    // Rebuild the binding name.
    auto entity_name = context.entity_names().Get(binding->entity_name_id);
    CARBON_CHECK(is_generic == entity_name.bind_index().has_value());
    auto entity_name_id = context.entity_names().AddSymbolicBindingName(
        entity_name.name_id, context.scope_stack().PeekNameScopeId(),
        is_generic ? context.scope_stack().AddCompileTimeBinding()
                   : SemIR::CompileTimeBindIndex::None,
        entity_name.is_template);

    // Rebuild the binding pattern.
    new_pattern_id = AddPatternInst(
        context,
        SemIR::LocIdAndInst::UncheckedLoc(
            SemIR::LocId(pattern_id),
            SemIR::AnyBindingPattern{.kind = binding->kind,
                                     .type_id = inner_type_id,
                                     .entity_name_id = entity_name_id}));

    // Also create and register the corresponding BindName instruction.
    auto bind_name_id = AddInstInNoBlock(
        context, SemIR::LocIdAndInst::UncheckedLoc(
                     SemIR::LocId(pattern_id),
                     SemIR::AnyBindName{
                         .kind = is_generic ? SemIR::BindSymbolicName::Kind
                                            : SemIR::BindName::Kind,
                         .type_id = inner_type_id,
                         .entity_name_id = entity_name_id,
                         .value_id = SemIR::InstId::None}));
    if (is_generic) {
      context.scope_stack().PushCompileTimeBinding(bind_name_id);
    }

    auto type_expr_region_id = context.sem_ir().expr_regions().Add(
        {.block_ids = {SemIR::InstBlockId::Empty},
         .result_id = context.types().GetInstId(inner_type_id)});
    bool inserted = context.bind_name_map()
                        .Insert(new_pattern_id,
                                {.bind_name_id = bind_name_id,
                                 .type_expr_region_id = type_expr_region_id})
                        .is_inserted();
    CARBON_CHECK(inserted);
  } else if (auto return_slot = pattern.TryAs<SemIR::ReturnSlotPattern>()) {
    new_pattern_id = AddPatternInst(
        context,
        SemIR::LocIdAndInst::UncheckedLoc(
            SemIR::LocId(pattern_id),
            SemIR::ReturnSlotPattern{.type_id = inner_type_id,
                                     .type_inst_id = SemIR::TypeInstId::None}));
  } else {
    CARBON_CHECK(pattern.Is<SemIR::ErrorInst>(),
                 "Unexpected pattern {0} in function signature", pattern);
    return SemIR::ErrorInst::InstId;
  }

  // Rebuild parameter.
  if (param) {
    auto type_id =
        SemIR::GetTypeOfInstInSpecific(context.sem_ir(), specific_id, param_id);
    new_pattern_id = AddPatternInst(
        context,
        SemIR::LocIdAndInst::UncheckedLoc(
            SemIR::LocId(param_id),
            SemIR::AnyParamPattern{.kind = param->kind,
                                   .type_id = type_id,
                                   .subpattern_id = new_pattern_id,
                                   .index = SemIR::CallParamIndex::None}));
  }

  // Rebuild `addr`.
  if (addr) {
    auto type_id =
        SemIR::GetTypeOfInstInSpecific(context.sem_ir(), specific_id, addr_id);
    new_pattern_id = AddPatternInst(
        context, SemIR::LocIdAndInst::UncheckedLoc(
                     SemIR::LocId(addr_id),
                     SemIR::AddrPattern{.type_id = type_id,
                                        .inner_id = new_pattern_id}));
  }

  return new_pattern_id;
}

static auto ClonePatternBlock(Context& context, SemIR::SpecificId specific_id,
                              SemIR::InstBlockId inst_block_id)
    -> SemIR::InstBlockId {
  if (!inst_block_id.has_value()) {
    return SemIR::InstBlockId::None;
  }
  auto orig_block = context.inst_blocks().Get(inst_block_id);

  llvm::SmallVector<SemIR::InstId> block;
  block.reserve(orig_block.size());
  for (auto inst_id : orig_block) {
    block.push_back(ClonePattern(context, specific_id, inst_id));
  }
  return context.inst_blocks().Add(block);
}

static auto CloneFunctionDecl(Context& context, SemIR::LocId loc_id,
                              SemIR::FunctionId signature_id,
                              SemIR::SpecificId signature_specific_id,
                              SemIR::FunctionId callee_id)
    -> std::pair<SemIR::FunctionId, SemIR::InstId> {
  StartGenericDecl(context);

  // Clone the signature. Note that we re-get the function after each of these,
  // because they might trigger imports that invalidate the function.
  context.pattern_block_stack().Push();
  auto implicit_param_patterns_id = ClonePatternBlock(
      context, signature_specific_id,
      context.functions().Get(signature_id).implicit_param_patterns_id);
  auto param_patterns_id = ClonePatternBlock(
      context, signature_specific_id,
      context.functions().Get(signature_id).param_patterns_id);
  auto return_slot_pattern_id = ClonePattern(
      context, signature_specific_id,
      context.functions().Get(signature_id).return_slot_pattern_id);
  auto self_param_id = FindSelfPattern(context, implicit_param_patterns_id);
  auto pattern_block_id = context.pattern_block_stack().Pop();

  // Perform callee-side pattern matching to rebuild the parameter list.
  context.inst_block_stack().Push();
  auto call_params_id =
      CalleePatternMatch(context, implicit_param_patterns_id, param_patterns_id,
                         return_slot_pattern_id);
  auto decl_block_id = context.inst_block_stack().Pop();

  // TODO: Create a ReturnSlot if needed, or switch to using CalleePatternMatch.

  // Create the `FunctionDecl` instruction.
  SemIR::FunctionDecl function_decl = {SemIR::TypeId::None,
                                       SemIR::FunctionId::None, decl_block_id};
  auto decl_id = AddPlaceholderInst(
      context, SemIR::LocIdAndInst::UncheckedLoc(loc_id, function_decl));
  auto generic_id = BuildGenericDecl(context, decl_id);

  // Create the `Function` object.
  auto& signature = context.functions().Get(signature_id);
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

// Build an expression that names the value matched by a pattern.
static auto BuildPatternRef(Context& context, SemIR::FunctionId function_id,
                            SemIR::InstId pattern_id) -> SemIR::InstId {
  CARBON_KIND_SWITCH(context.insts().Get(pattern_id)) {
    case CARBON_KIND(SemIR::ValueParamPattern value_param): {
      // Build a reference to this parameter.
      auto call_param_id = context.inst_blocks().Get(
          context.functions()
              .Get(function_id)
              .call_params_id)[value_param.index.index];
      // Use a pretty name for the `name_ref`. While it's suspicious to use a
      // pretty name in the IR like this, the only reason we include a name at
      // all here is to make the formatted SemIR more readable.
      return AddInst<SemIR::NameRef>(
          context, SemIR::LocId(pattern_id),
          {.type_id = context.insts().Get(call_param_id).type_id(),
           .name_id = SemIR::GetPrettyNameFromPatternId(
               context.sem_ir(), value_param.subpattern_id),
           .value_id = call_param_id});
    }

    case CARBON_KIND(SemIR::AddrPattern addr): {
      // TODO: Make non-recursive.
      auto ptr_id = BuildPatternRef(context, function_id, addr.inner_id);
      return PerformPointerDereference(
          context, SemIR::LocId(pattern_id), ptr_id, [](SemIR::TypeId) {
            CARBON_FATAL("addr subpattern is not a pointer");
          });
    }

    case SemIR::ErrorInst::Kind: {
      return SemIR::ErrorInst::InstId;
    }

    default: {
      context.TODO(
          pattern_id,
          "don't know how to build reference to this pattern in thunk");
      return SemIR::ErrorInst::InstId;
    }
  }
}

// Build a call to a function that forwards the arguments of the enclosing
// function, for use when constructing a thunk.
static auto BuildThunkCall(Context& context, SemIR::FunctionId function_id,
                           SemIR::InstId callee_id) -> SemIR::InstId {
  auto loc_id = SemIR::LocId(callee_id);
  auto& function = context.functions().Get(function_id);

  // If we have a self parameter, form `self.<callee_id>`.
  if (function.self_param_id.has_value()) {
    callee_id = PerformCompoundMemberAccess(
        context, loc_id, BuildPatternRef(context, function_id, function.self_param_id),
        callee_id);
  }

  // Form an argument list.
  llvm::SmallVector<SemIR::InstId> args;
  for (auto pattern_id :
       context.inst_blocks().Get(function.param_patterns_id)) {
    args.push_back(BuildPatternRef(context, function_id, pattern_id));
  }

  return PerformCall(context, loc_id, callee_id, args);
}

static auto HasDeclaredReturnType(Context& context,
                                  SemIR::FunctionId function_id) -> bool {
  return context.functions()
      .Get(function_id)
      .return_slot_pattern_id.has_value();
}

static auto BuildThunkDefinition(Context& context,
                                 SemIR::FunctionId function_id,
                                 SemIR::InstId thunk_id,
                                 SemIR::InstId callee_id) -> bool {
  // Suppress diagnostics produced when building the thunk.
  bool any_errors = false;
  Diagnostics::AnnotationScope annot_scope(
      &context.emitter(), [&](DiagnosticBuilder& builder) {
        // Suppress any diagnostics being produced while building the thunk.
        //
        // TODO: Do an up-front check of whether the thunk obviously won't work,
        // and diagnose that if not, but otherwise produce the diagnostics we
        // get when building the thunk.
        builder = context.emitter().BuildSuppressed();

        // TODO: Distinguish errors from warnings.
        any_errors = true;
      });

  // TODO: This duplicates much of the handling for FunctionDefinitionStart and
  // FunctionDefinition parse nodes. Consider refactoring.
  context.scope_stack().PushForFunctionBody(thunk_id);
  context.inst_block_stack().Push();
  context.region_stack().PushRegion(context.inst_block_stack().PeekOrAdd());
  StartGenericDefinition(context);

  auto call_id = BuildThunkCall(context, function_id, callee_id);
  if (HasDeclaredReturnType(context, function_id)) {
    BuildReturnWithExpr(context, SemIR::LocId(callee_id), call_id);
  } else {
    BuildReturnWithNoExpr(context, SemIR::LocId(callee_id));
  }

  context.inst_block_stack().Pop();
  context.scope_stack().Pop();

  auto& function = context.functions().Get(function_id);
  function.body_block_ids = context.region_stack().PopRegion();
  FinishGenericDefinition(context, function.generic_id);

  return !any_errors;
}

auto BuildThunk(Context& context, SemIR::FunctionId signature_id,
                SemIR::SpecificId signature_specific_id,
                SemIR::InstId callee_id) -> SemIR::InstId {
  auto callee = SemIR::GetCalleeFunction(context.sem_ir(), callee_id);

  // Check whether we can use the given function without a thunk.
  // TODO: For virtual functions, we want different rules for checking `self`.
  if (CheckFunctionTypeMatches(
          context, context.functions().Get(callee.function_id),
          context.functions().Get(signature_id), signature_specific_id,
          /*check_syntax=*/false, /*check_self=*/true, /*diagnose=*/false)) {
    return callee_id;
  }

  // Create a scope for the function's parameters and generic parameters.
  context.scope_stack().PushForDeclName();

  // We can't use the function directly. Build a thunk.
  // TODO: Check for and diagnose obvious reasons why this will fail, such as
  // arity mismatch, before trying to build the thunk.
  auto [function_id, thunk_id] = CloneFunctionDecl(
      context, SemIR::LocId(callee_id), signature_id,
      signature_specific_id, callee.function_id);

  // TODO: We need to delay doing this until we get to the end of the enclosing
  // deferred definition scope, if there is one. For example, an `impl` inside a
  // `class` definition should have its thunks defined at the end of the class,
  // like they would be if they were defined inline.
  if (!BuildThunkDefinition(context, function_id, thunk_id, callee_id)) {
    // If building the thunk failed, produce the basic type mismatch diagnostic.
    bool success = CheckFunctionTypeMatches(
        context, context.functions().Get(callee.function_id),
        context.functions().Get(signature_id), signature_specific_id,
        /*check_syntax=*/false, /*check_self=*/true);
    CARBON_CHECK(!success, "Function type unexpectedly started to match");
  } else if (!HasDeclaredReturnType(context, signature_id) &&
             HasDeclaredReturnType(context, callee.function_id)) {
    // P3763:
    //   If the function in the interface does not have a return type, the
    //   program is invalid if the function in the impl specifies a return type.
    //
    // Call into the redeclaration checking logic to produce a suitable error.
    bool success = CheckFunctionReturnTypeMatches(
        context, context.functions().Get(callee.function_id),
        context.functions().Get(signature_id), signature_specific_id);
    CARBON_CHECK(!success, "Return type unexpectedly matches");
  }

  context.scope_stack().Pop();

  return thunk_id;
}

}  // namespace Carbon::Check
