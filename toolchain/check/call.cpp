// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/call.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/deduce.h"
#include "toolchain/check/function.h"
#include "toolchain/check/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Performs a call where the callee is the name of a generic class, such as
// `Vector(i32)`.
static auto PerformCallToGenericClass(Context& context, Parse::NodeId node_id,
                                      SemIR::InstId callee_id,
                                      SemIR::ClassId class_id,
                                      llvm::ArrayRef<SemIR::InstId> arg_ids)
    -> SemIR::InstId {
  auto& class_info = context.classes().Get(class_id);

  // TODO: Pass in information about the specific in which the generic class
  // name was found.
  // TODO: Perform argument deduction.
  auto specific_id = SemIR::SpecificId::Invalid;

  // Convert the arguments to match the parameters.
  auto converted_args_id = ConvertCallArgs(
      context, node_id, /*self_id=*/SemIR::InstId::Invalid, arg_ids,
      /*return_storage_id=*/SemIR::InstId::Invalid, class_info, specific_id);
  return context.AddInst<SemIR::Call>(node_id,
                                      {.type_id = SemIR::TypeId::TypeType,
                                       .callee_id = callee_id,
                                       .args_id = converted_args_id});
}

// Performs a call where the callee is the name of a generic interface, such as
// `AddWith(i32)`.
// TODO: Refactor with PerformCallToGenericClass.
static auto PerformCallToGenericInterface(Context& context,
                                          Parse::NodeId node_id,
                                          SemIR::InstId callee_id,
                                          SemIR::InterfaceId interface_id,
                                          llvm::ArrayRef<SemIR::InstId> arg_ids)
    -> SemIR::InstId {
  auto& interface_info = context.interfaces().Get(interface_id);

  // TODO: Pass in information about the specific in which the generic interface
  // name was found.
  // TODO: Perform argument deduction.
  auto specific_id = SemIR::SpecificId::Invalid;

  // Convert the arguments to match the parameters.
  auto converted_args_id = ConvertCallArgs(
      context, node_id, /*self_id=*/SemIR::InstId::Invalid, arg_ids,
      /*return_storage_id=*/SemIR::InstId::Invalid, interface_info,
      specific_id);
  return context.AddInst<SemIR::Call>(node_id,
                                      {.type_id = SemIR::TypeId::TypeType,
                                       .callee_id = callee_id,
                                       .args_id = converted_args_id});
}

auto PerformCall(Context& context, Parse::NodeId node_id,
                 SemIR::InstId callee_id, llvm::ArrayRef<SemIR::InstId> arg_ids)
    -> SemIR::InstId {
  // Identify the function we're calling.
  auto callee_function = GetCalleeFunction(context.sem_ir(), callee_id);
  if (!callee_function.function_id.is_valid()) {
    auto type_inst =
        context.types().GetAsInst(context.insts().Get(callee_id).type_id());
    CARBON_KIND_SWITCH(type_inst) {
      case CARBON_KIND(SemIR::GenericClassType generic_class): {
        return PerformCallToGenericClass(context, node_id, callee_id,
                                         generic_class.class_id, arg_ids);
      }
      case CARBON_KIND(SemIR::GenericInterfaceType generic_interface): {
        return PerformCallToGenericInterface(context, node_id, callee_id,
                                             generic_interface.interface_id,
                                             arg_ids);
      }
      default: {
        if (!callee_function.is_error) {
          CARBON_DIAGNOSTIC(CallToNonCallable, Error,
                            "Value of type `{0}` is not callable.",
                            SemIR::TypeId);
          context.emitter().Emit(node_id, CallToNonCallable,
                                 context.insts().Get(callee_id).type_id());
        }
        return SemIR::InstId::BuiltinError;
      }
    }
  }
  auto& callable = context.functions().Get(callee_function.function_id);

  // If the callee is a generic function, determine the generic argument values
  // for the call.
  auto specific_id = SemIR::SpecificId::Invalid;
  if (callable.generic_id.is_valid()) {
    specific_id = DeduceGenericCallArguments(
        context, node_id, callable.generic_id, callee_function.specific_id,
        callable.implicit_param_refs_id, callable.param_refs_id,
        callee_function.self_id, arg_ids);
    if (!specific_id.is_valid()) {
      return SemIR::InstId::BuiltinError;
    }
  }

  // If there is a return slot, build storage for the result.
  SemIR::InstId return_storage_id = SemIR::InstId::Invalid;
  SemIR::ReturnTypeInfo return_info = [&] {
    DiagnosticAnnotationScope annotate_diagnostics(
        &context.emitter(), [&](auto& builder) {
          CARBON_DIAGNOSTIC(IncompleteReturnTypeHere, Note,
                            "Return type declared here.");
          builder.Note(callable.return_storage_id, IncompleteReturnTypeHere);
        });
    return CheckFunctionReturnType(context, callee_id, callable, specific_id);
  }();
  switch (return_info.init_repr.kind) {
    case SemIR::InitRepr::InPlace:
      // Tentatively put storage for a temporary in the function's return slot.
      // This will be replaced if necessary when we perform initialization.
      return_storage_id = context.AddInst<SemIR::TemporaryStorage>(
          node_id, {.type_id = return_info.type_id});
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
  auto converted_args_id =
      ConvertCallArgs(context, node_id, callee_function.self_id, arg_ids,
                      return_storage_id, callable, specific_id);
  auto call_inst_id =
      context.AddInst<SemIR::Call>(node_id, {.type_id = return_info.type_id,
                                             .callee_id = callee_id,
                                             .args_id = converted_args_id});

  return call_inst_id;
}

}  // namespace Carbon::Check
