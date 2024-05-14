// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/call.h"

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/function.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Performs a call where the callee is the name of a generic class, such as
// `Vector(i32)`.
static auto PerformCallToGenericClass(Context& context, Parse::NodeId node_id,
                                      SemIR::ClassId class_id,
                                      llvm::ArrayRef<SemIR::InstId> arg_ids)
    -> SemIR::InstId {
  auto& class_info = context.classes().Get(class_id);

  // Convert the arguments to match the parameters.
  auto converted_args_id = ConvertCallArgs(
      context, node_id, /*self_id=*/SemIR::InstId::Invalid, arg_ids,
      /*return_storage_id=*/SemIR::InstId::Invalid, class_info.decl_id,
      class_info.implicit_param_refs_id, class_info.param_refs_id);
  return context.AddInst(
      {node_id,
       SemIR::ClassType{SemIR::TypeId::TypeType, class_id, converted_args_id}});
}

auto PerformCall(Context& context, Parse::NodeId node_id,
                 SemIR::InstId callee_id, llvm::ArrayRef<SemIR::InstId> arg_ids)
    -> SemIR::InstId {
  // Identify the function we're calling.
  auto callee_function = GetCalleeFunction(context.sem_ir(), callee_id);
  if (!callee_function.function_id.is_valid()) {
    if (auto generic_class = context.types().TryGetAs<SemIR::GenericClassType>(
            context.insts().Get(callee_id).type_id())) {
      return PerformCallToGenericClass(context, node_id,
                                       generic_class->class_id, arg_ids);
    }
    if (!callee_function.is_error) {
      CARBON_DIAGNOSTIC(CallToNonCallable, Error,
                        "Value of type `{0}` is not callable.", SemIR::TypeId);
      context.emitter().Emit(node_id, CallToNonCallable,
                             context.insts().Get(callee_id).type_id());
    }
    return SemIR::InstId::BuiltinError;
  }
  auto& callable = context.functions().Get(callee_function.function_id);

  // For functions with an implicit return type, the return type is the empty
  // tuple type.
  SemIR::TypeId type_id = callable.return_type_id;
  if (!type_id.is_valid()) {
    type_id = context.GetTupleType({});
  }

  // If there is a return slot, build storage for the result.
  SemIR::InstId return_storage_id = SemIR::InstId::Invalid;
  {
    DiagnosticAnnotationScope annotate_diagnostics(
        &context.emitter(), [&](auto& builder) {
          CARBON_DIAGNOSTIC(IncompleteReturnTypeHere, Note,
                            "Return type declared here.");
          builder.Note(callable.return_storage_id, IncompleteReturnTypeHere);
        });
    CheckFunctionReturnType(context, callee_id, callable);
  }
  switch (callable.return_slot) {
    case SemIR::Function::ReturnSlot::Present:
      // Tentatively put storage for a temporary in the function's return slot.
      // This will be replaced if necessary when we perform initialization.
      return_storage_id = context.AddInst(
          {node_id, SemIR::TemporaryStorage{callable.return_type_id}});
      break;
    case SemIR::Function::ReturnSlot::Absent:
      break;
    case SemIR::Function::ReturnSlot::Error:
      // Don't form an initializing expression with an incomplete type.
      type_id = SemIR::TypeId::Error;
      break;
    case SemIR::Function::ReturnSlot::NotComputed:
      CARBON_FATAL() << "Missing return slot category in call to " << callable;
  }

  // Convert the arguments to match the parameters.
  auto converted_args_id =
      ConvertCallArgs(context, node_id, callee_function.self_id, arg_ids,
                      return_storage_id, callable.decl_id,
                      callable.implicit_param_refs_id, callable.param_refs_id);
  auto call_inst_id = context.AddInst(
      {node_id, SemIR::Call{type_id, callee_id, converted_args_id}});

  return call_inst_id;
}

}  // namespace Carbon::Check
