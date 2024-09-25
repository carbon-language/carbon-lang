// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/call.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/deduce.h"
#include "toolchain/check/function.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Resolves the callee expression in a call to a specific callee, or diagnoses
// if no specific callee can be identified. This verifies the arity of the
// callee and determines any compile-time arguments, but doesn't check that the
// runtime arguments are convertible to the parameter types.
//
// `self_id` and `arg_ids` are the self argument and explicit arguments in the
// call.
//
// Returns a SpecificId for the specific callee, or `nullopt` if an error has
// been diagnosed.
static auto ResolveCalleeInCall(Context& context, SemIR::LocId loc_id,
                                const SemIR::EntityWithParamsBase& entity,
                                llvm::StringLiteral entity_kind_for_diagnostic,
                                SemIR::GenericId entity_generic_id,
                                SemIR::SpecificId enclosing_specific_id,
                                SemIR::InstId self_id,
                                llvm::ArrayRef<SemIR::InstId> arg_ids)
    -> std::optional<SemIR::SpecificId> {
  CalleeParamsInfo callee_info(entity);

  // Check that the arity matches.
  auto params = context.inst_blocks().GetOrEmpty(callee_info.param_refs_id);
  if (arg_ids.size() != params.size()) {
    CARBON_DIAGNOSTIC(CallArgCountMismatch, Error,
                      "{0} argument(s) passed to {1} expecting "
                      "{2} argument(s).",
                      int, llvm::StringLiteral, int);
    CARBON_DIAGNOSTIC(InCallToEntity, Note, "calling {0} declared here",
                      llvm::StringLiteral);
    context.emitter()
        .Build(loc_id, CallArgCountMismatch, arg_ids.size(),
               entity_kind_for_diagnostic, params.size())
        .Note(callee_info.callee_loc, InCallToEntity,
              entity_kind_for_diagnostic)
        .Emit();
    return std::nullopt;
  }

  // Perform argument deduction.
  auto specific_id = SemIR::SpecificId::Invalid;
  if (entity_generic_id.is_valid()) {
    specific_id = DeduceGenericCallArguments(
        context, loc_id, entity_generic_id, enclosing_specific_id,
        callee_info.implicit_param_refs_id, callee_info.param_refs_id, self_id,
        arg_ids);
    if (!specific_id.is_valid()) {
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
  auto callee_specific_id = ResolveCalleeInCall(
      context, loc_id, generic_class, "generic class", generic_class.generic_id,
      enclosing_specific_id, /*self_id=*/SemIR::InstId::Invalid, arg_ids);
  if (!callee_specific_id) {
    return SemIR::InstId::BuiltinError;
  }
  return context.AddInst<SemIR::ClassType>(
      loc_id, {.type_id = SemIR::TypeId::TypeType,
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
  auto callee_specific_id = ResolveCalleeInCall(
      context, loc_id, interface, "generic interface", interface.generic_id,
      enclosing_specific_id, /*self_id=*/SemIR::InstId::Invalid, arg_ids);
  if (!callee_specific_id) {
    return SemIR::InstId::BuiltinError;
  }
  return context.AddInst<SemIR::InterfaceType>(
      loc_id, {.type_id = SemIR::TypeId::TypeType,
               .interface_id = interface_id,
               .specific_id = *callee_specific_id});
}

auto PerformCall(Context& context, SemIR::LocId loc_id, SemIR::InstId callee_id,
                 llvm::ArrayRef<SemIR::InstId> arg_ids) -> SemIR::InstId {
  // Identify the function we're calling.
  auto callee_function = GetCalleeFunction(context.sem_ir(), callee_id);
  if (!callee_function.function_id.is_valid()) {
    auto type_inst =
        context.types().GetAsInst(context.insts().Get(callee_id).type_id());
    CARBON_KIND_SWITCH(type_inst) {
      case CARBON_KIND(SemIR::GenericClassType generic_class): {
        return PerformCallToGenericClass(
            context, loc_id, generic_class.class_id,
            generic_class.enclosing_specific_id, arg_ids);
      }
      case CARBON_KIND(SemIR::GenericInterfaceType generic_interface): {
        return PerformCallToGenericInterface(
            context, loc_id, generic_interface.interface_id,
            generic_interface.enclosing_specific_id, arg_ids);
      }
      default: {
        if (!callee_function.is_error) {
          CARBON_DIAGNOSTIC(CallToNonCallable, Error,
                            "value of type `{0}` is not callable",
                            SemIR::TypeId);
          context.emitter().Emit(loc_id, CallToNonCallable,
                                 context.insts().Get(callee_id).type_id());
        }
        return SemIR::InstId::BuiltinError;
      }
    }
  }
  auto& callable = context.functions().Get(callee_function.function_id);

  // If the callee is a generic function, determine the generic argument values
  // for the call.
  auto callee_specific_id = ResolveCalleeInCall(
      context, loc_id, callable, "function", callable.generic_id,
      callee_function.specific_id, callee_function.self_id, arg_ids);
  if (!callee_specific_id) {
    return SemIR::InstId::BuiltinError;
  }

  // If there is a return slot, build storage for the result.
  SemIR::InstId return_storage_id = SemIR::InstId::Invalid;
  SemIR::ReturnTypeInfo return_info = [&] {
    DiagnosticAnnotationScope annotate_diagnostics(
        &context.emitter(), [&](auto& builder) {
          CARBON_DIAGNOSTIC(IncompleteReturnTypeHere, Note,
                            "return type declared here");
          builder.Note(callable.return_storage_id, IncompleteReturnTypeHere);
        });
    return CheckFunctionReturnType(context, callee_id, callable,
                                   *callee_specific_id);
  }();
  switch (return_info.init_repr.kind) {
    case SemIR::InitRepr::InPlace:
      // Tentatively put storage for a temporary in the function's return slot.
      // This will be replaced if necessary when we perform initialization.
      return_storage_id = context.AddInst<SemIR::TemporaryStorage>(
          loc_id, {.type_id = return_info.type_id});
      break;
    case SemIR::InitRepr::None:
      // For functions with an implicit return type, the return type is the
      // empty tuple type.
      if (!return_info.type_id.is_valid()) {
        return_info.type_id = context.GetTupleType({});
      }
      break;
    case SemIR::InitRepr::ByCopy:
      break;
    case SemIR::InitRepr::Incomplete:
      // Don't form an initializing expression with an incomplete type.
      // CheckFunctionReturnType will have diagnosed this for us if needed.
      return_info.type_id = SemIR::TypeId::Error;
      break;
  }

  // Convert the arguments to match the parameters.
  auto converted_args_id = ConvertCallArgs(
      context, loc_id, callee_function.self_id, arg_ids, return_storage_id,
      CalleeParamsInfo(callable), *callee_specific_id);
  auto call_inst_id =
      context.AddInst<SemIR::Call>(loc_id, {.type_id = return_info.type_id,
                                            .callee_id = callee_id,
                                            .args_id = converted_args_id});

  return call_inst_id;
}

}  // namespace Carbon::Check
