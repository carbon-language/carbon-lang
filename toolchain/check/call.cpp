// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/call.h"

#include <optional>

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/control_flow.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/cpp_overload_resolution.h"
#include "toolchain/check/cpp_thunk.h"
#include "toolchain/check/deduce.h"
#include "toolchain/check/facet_type.h"
#include "toolchain/check/function.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/name_ref.h"
#include "toolchain/check/thunk.h"
#include "toolchain/check/type.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

namespace {
// Entity kinds, for diagnostics. Converted to an int for a select.
enum class EntityKind : uint8_t {
  Function = 0,
  GenericClass = 1,
  GenericInterface = 2,
};
}  // namespace

// Resolves the callee expression in a call to a specific callee, or diagnoses
// if no specific callee can be identified. This verifies the arity of the
// callee and determines any compile-time arguments, but doesn't check that the
// runtime arguments are convertible to the parameter types.
//
// `self_id` and `arg_ids` are the self argument and explicit arguments in the
// call.
//
// Returns a `SpecificId` for the specific callee, `SpecificId::None` if the
// callee is not generic, or `nullopt` if an error has been diagnosed.
static auto ResolveCalleeInCall(Context& context, SemIR::LocId loc_id,
                                const SemIR::EntityWithParamsBase& entity,
                                EntityKind entity_kind_for_diagnostic,
                                SemIR::SpecificId enclosing_specific_id,
                                SemIR::InstId self_type_id,
                                SemIR::InstId self_id,
                                llvm::ArrayRef<SemIR::InstId> arg_ids)
    -> std::optional<SemIR::SpecificId> {
  // Check that the arity matches.
  auto params = context.inst_blocks().GetOrEmpty(entity.param_patterns_id);
  if (arg_ids.size() != params.size()) {
    CARBON_DIAGNOSTIC(CallArgCountMismatch, Error,
                      "{0} argument{0:s} passed to "
                      "{1:=0:function|=1:generic class|=2:generic interface}"
                      " expecting {2} argument{2:s}",
                      Diagnostics::IntAsSelect, Diagnostics::IntAsSelect,
                      Diagnostics::IntAsSelect);
    CARBON_DIAGNOSTIC(
        InCallToEntity, Note,
        "calling {0:=0:function|=1:generic class|=2:generic interface}"
        " declared here",
        Diagnostics::IntAsSelect);
    context.emitter()
        .Build(loc_id, CallArgCountMismatch, arg_ids.size(),
               static_cast<int>(entity_kind_for_diagnostic), params.size())
        .Note(entity.latest_decl_id(), InCallToEntity,
              static_cast<int>(entity_kind_for_diagnostic))
        .Emit();
    return std::nullopt;
  }

  // Perform argument deduction.
  auto specific_id = SemIR::SpecificId::None;
  if (entity.generic_id.has_value()) {
    specific_id = DeduceGenericCallArguments(
        context, loc_id, entity.generic_id, enclosing_specific_id, self_type_id,
        entity.implicit_param_patterns_id, entity.param_patterns_id, self_id,
        arg_ids);
    if (!specific_id.has_value()) {
      return std::nullopt;
    }
  }
  return specific_id;
}

// Performs a call where the callee is the name of a generic class, such as
// `Vector(i32)`.
static auto PerformCallToGenericClass(Context& context, SemIR::LocId loc_id,
                                      SemIR::ClassId class_id,
                                      SemIR::SpecificId enclosing_specific_id,
                                      llvm::ArrayRef<SemIR::InstId> arg_ids)
    -> SemIR::InstId {
  const auto& generic_class = context.classes().Get(class_id);
  auto callee_specific_id =
      ResolveCalleeInCall(context, loc_id, generic_class,
                          EntityKind::GenericClass, enclosing_specific_id,
                          /*self_type_id=*/SemIR::InstId::None,
                          /*self_id=*/SemIR::InstId::None, arg_ids);
  if (!callee_specific_id) {
    return SemIR::ErrorInst::InstId;
  }
  return GetOrAddInst<SemIR::ClassType>(context, loc_id,
                                        {.type_id = SemIR::TypeType::TypeId,
                                         .class_id = class_id,
                                         .specific_id = *callee_specific_id});
}

// Performs a call where the callee is the name of a generic interface, such as
// `AddWith(i32)`.
static auto PerformCallToGenericInterface(
    Context& context, SemIR::LocId loc_id, SemIR::InterfaceId interface_id,
    SemIR::SpecificId enclosing_specific_id,
    llvm::ArrayRef<SemIR::InstId> arg_ids) -> SemIR::InstId {
  const auto& interface = context.interfaces().Get(interface_id);
  auto callee_specific_id =
      ResolveCalleeInCall(context, loc_id, interface,
                          EntityKind::GenericInterface, enclosing_specific_id,
                          /*self_type_id=*/SemIR::InstId::None,
                          /*self_id=*/SemIR::InstId::None, arg_ids);
  if (!callee_specific_id) {
    return SemIR::ErrorInst::InstId;
  }
  return GetOrAddInst(
      context, loc_id,
      FacetTypeFromInterface(context, interface_id, *callee_specific_id));
}

// Builds an appropriate specific function for the callee, also handling
// instance binding.
static auto BuildCalleeSpecificFunction(
    Context& context, SemIR::LocId loc_id, SemIR::InstId callee_id,
    SemIR::InstId callee_function_self_type_id,
    SemIR::SpecificId callee_specific_id) -> SemIR::InstId {
  auto generic_callee_id = callee_id;

  // Strip off a bound_method so that we can form a constant specific callee.
  auto bound_method = context.insts().TryGetAs<SemIR::BoundMethod>(callee_id);
  if (bound_method) {
    generic_callee_id = bound_method->function_decl_id;
  }

  // Form a specific callee.
  if (callee_function_self_type_id.has_value()) {
    // This is an associated function in an interface; the callee is the
    // specific function in the impl that corresponds to the specific function
    // we deduced.
    callee_id =
        GetOrAddInst(context, SemIR::LocId(generic_callee_id),
                     SemIR::SpecificImplFunction{
                         .type_id = GetSingletonType(
                             context, SemIR::SpecificFunctionType::TypeInstId),
                         .callee_id = generic_callee_id,
                         .specific_id = callee_specific_id});
  } else {
    // This is a regular generic function. The callee is the specific function
    // we deduced.
    callee_id =
        GetOrAddInst(context, SemIR::LocId(generic_callee_id),
                     SemIR::SpecificFunction{
                         .type_id = GetSingletonType(
                             context, SemIR::SpecificFunctionType::TypeInstId),
                         .callee_id = generic_callee_id,
                         .specific_id = callee_specific_id});
  }

  // Add the `self` argument back if there was one.
  if (bound_method) {
    callee_id =
        GetOrAddInst<SemIR::BoundMethod>(context, loc_id,
                                         {.type_id = bound_method->type_id,
                                          .object_id = bound_method->object_id,
                                          .function_decl_id = callee_id});
  }

  return callee_id;
}

// Returns the return type, with a scoped annotation for any diagnostics.
static auto CheckCalleeFunctionReturnType(Context& context, SemIR::LocId loc_id,
                                          SemIR::FunctionId callee_function_id,
                                          SemIR::SpecificId callee_specific_id)
    -> SemIR::ReturnTypeInfo {
  auto& function = context.functions().Get(callee_function_id);
  Diagnostics::AnnotationScope annotate_diagnostics(
      &context.emitter(), [&](auto& builder) {
        CARBON_DIAGNOSTIC(IncompleteReturnTypeHere, Note,
                          "return type declared here");
        builder.Note(function.return_slot_pattern_id, IncompleteReturnTypeHere);
      });
  return CheckFunctionReturnType(context, loc_id, function, callee_specific_id);
}

// Performs a call where the callee is a function.
static auto PerformCallToFunction(Context& context, SemIR::LocId loc_id,
                                  SemIR::InstId callee_id,
                                  const SemIR::CalleeFunction& callee_function,
                                  llvm::ArrayRef<SemIR::InstId> arg_ids)
    -> SemIR::InstId {
  // If the callee is a generic function, determine the generic argument values
  // for the call.
  auto callee_specific_id = ResolveCalleeInCall(
      context, loc_id, context.functions().Get(callee_function.function_id),
      EntityKind::Function, callee_function.enclosing_specific_id,
      callee_function.self_type_id, callee_function.self_id, arg_ids);
  if (!callee_specific_id) {
    return SemIR::ErrorInst::InstId;
  }

  if (callee_specific_id->has_value()) {
    callee_id = BuildCalleeSpecificFunction(context, loc_id, callee_id,
                                            callee_function.self_type_id,
                                            *callee_specific_id);
  }

  // If there is a return slot, build storage for the result.
  SemIR::ReturnTypeInfo return_info = CheckCalleeFunctionReturnType(
      context, loc_id, callee_function.function_id, *callee_specific_id);
  SemIR::InstId return_slot_arg_id = SemIR::InstId::None;
  switch (return_info.init_repr.kind) {
    case SemIR::InitRepr::InPlace:
      // Tentatively put storage for a temporary in the function's return slot.
      // This will be replaced if necessary when we perform initialization.
      return_slot_arg_id = AddInst<SemIR::TemporaryStorage>(
          context, loc_id, {.type_id = return_info.type_id});
      break;
    case SemIR::InitRepr::None:
      // For functions with an implicit return type, the return type is the
      // empty tuple type.
      if (!return_info.type_id.has_value()) {
        return_info.type_id = GetTupleType(context, {});
      }
      break;
    case SemIR::InitRepr::ByCopy:
      break;
    case SemIR::InitRepr::Incomplete:
      // Don't form an initializing expression with an incomplete type.
      // CheckFunctionReturnType will have diagnosed this for us if needed.
      return_info.type_id = SemIR::ErrorInst::TypeId;
      break;
  }

  auto& callee = context.functions().Get(callee_function.function_id);

  // Convert the arguments to match the parameters.
  auto converted_args_id =
      ConvertCallArgs(context, loc_id, callee_function.self_id, arg_ids,
                      return_slot_arg_id, callee, *callee_specific_id);

  switch (callee.special_function_kind) {
    case SemIR::Function::SpecialFunctionKind::Thunk: {
      // If we're about to form a direct call to a thunk, inline it.
      LoadImportRef(context, callee.thunk_decl_id());

      // Name the thunk target within the enclosing scope of the thunk.
      auto thunk_ref_id =
          BuildNameRef(context, loc_id, callee.name_id, callee.thunk_decl_id(),
                       callee_function.enclosing_specific_id);

      // This recurses back into `PerformCall`. However, we never form a thunk
      // to a thunk, so we only recurse once.
      return PerformThunkCall(context, loc_id, callee_function.function_id,
                              context.inst_blocks().Get(converted_args_id),
                              thunk_ref_id);
    }

    case SemIR::Function::SpecialFunctionKind::HasCppThunk: {
      // This recurses back into `PerformCall`. However, we never form a C++
      // thunk to a C++ thunk, so we only recurse once.
      return PerformCppThunkCall(context, loc_id, callee_function.function_id,
                                 context.inst_blocks().Get(converted_args_id),
                                 callee.cpp_thunk_decl_id());
    }

    case SemIR::Function::SpecialFunctionKind::None:
    case SemIR::Function::SpecialFunctionKind::Builtin: {
      return GetOrAddInst<SemIR::Call>(context, loc_id,
                                       {.type_id = return_info.type_id,
                                        .callee_id = callee_id,
                                        .args_id = converted_args_id});
    }
  }
}

auto PerformCall(Context& context, SemIR::LocId loc_id, SemIR::InstId callee_id,
                 llvm::ArrayRef<SemIR::InstId> arg_ids) -> SemIR::InstId {
  // Try treating the callee as a function first.
  auto callee_function = GetCalleeFunction(context.sem_ir(), callee_id);
  if (callee_function.is_error) {
    return SemIR::ErrorInst::InstId;
  }
  if (callee_function.cpp_overload_set_id.has_value()) {
    auto resolved_fn_id = PerformCppOverloadResolution(
        context, loc_id, callee_function.cpp_overload_set_id, arg_ids);
    if (!resolved_fn_id) {
      return SemIR::ErrorInst::InstId;
    }
    callee_id = resolved_fn_id.value();
    callee_function = GetCalleeFunction(context.sem_ir(), callee_id);
    if (callee_function.is_error) {
      return SemIR::ErrorInst::InstId;
    }
    CARBON_CHECK(!callee_function.cpp_overload_set_id.has_value());
  }
  if (callee_function.function_id.has_value()) {
    return PerformCallToFunction(context, loc_id, callee_id, callee_function,
                                 arg_ids);
  }

  // Callee isn't a function, so try treating it as a generic type.
  auto type_inst =
      context.types().GetAsInst(context.insts().Get(callee_id).type_id());
  CARBON_KIND_SWITCH(type_inst) {
    case CARBON_KIND(SemIR::GenericClassType generic_class): {
      return PerformCallToGenericClass(context, loc_id, generic_class.class_id,
                                       generic_class.enclosing_specific_id,
                                       arg_ids);
    }
    case CARBON_KIND(SemIR::GenericInterfaceType generic_interface): {
      return PerformCallToGenericInterface(
          context, loc_id, generic_interface.interface_id,
          generic_interface.enclosing_specific_id, arg_ids);
    }
    default: {
      CARBON_DIAGNOSTIC(CallToNonCallable, Error,
                        "value of type {0} is not callable", TypeOfInstId);
      context.emitter().Emit(loc_id, CallToNonCallable, callee_id);
      return SemIR::ErrorInst::InstId;
    }
  }
}

}  // namespace Carbon::Check
