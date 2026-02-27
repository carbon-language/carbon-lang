// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/function.h"

#include <optional>
#include <variant>

#include "toolchain/base/kind_switch.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

auto GetCallee(const File& sem_ir, InstId callee_id,
               SpecificId caller_specific_id) -> Callee {
  CalleeFunction fn = {.function_id = FunctionId::None,
                       .enclosing_specific_id = SpecificId::None,
                       .resolved_specific_id = SpecificId::None,
                       .self_type_id = InstId::None,
                       .self_id = InstId::None};
  if (auto bound_method = sem_ir.insts().TryGetAs<BoundMethod>(callee_id)) {
    fn.self_id = bound_method->object_id;
    callee_id = bound_method->function_decl_id;
  }

  if (caller_specific_id.has_value()) {
    callee_id = sem_ir.constant_values().GetInstIdIfValid(
        GetConstantValueInSpecific(sem_ir, caller_specific_id, callee_id));
    CARBON_CHECK(callee_id.has_value(),
                 "Invalid callee id in a specific context");
  }

  auto val_id = sem_ir.constant_values().GetConstantInstId(callee_id);
  if (!val_id.has_value()) {
    return CalleeNonFunction();
  }

  if (auto specific_function =
          sem_ir.insts().TryGetAs<SpecificFunction>(val_id)) {
    fn.resolved_specific_id = specific_function->specific_id;
    val_id = sem_ir.constant_values().GetConstantInstId(
        specific_function->callee_id);
  } else if (auto specific_impl_function =
                 sem_ir.insts().TryGetAs<SpecificImplFunction>(val_id)) {
    fn.resolved_specific_id = specific_impl_function->specific_id;
    val_id = sem_ir.constant_values().GetConstantInstId(
        specific_impl_function->callee_id);
  }
  if (!val_id.has_value()) {
    return CalleeNonFunction();
  }

  // Identify the function we're calling by its type.
  auto fn_type_inst_id =
      sem_ir.types().GetTypeInstId(sem_ir.insts().Get(val_id).type_id());

  if (auto cpp_overload_set_type =
          sem_ir.insts().TryGetAs<CppOverloadSetType>(fn_type_inst_id)) {
    CARBON_CHECK(!fn.resolved_specific_id.has_value(),
                 "Only `SpecificFunction` will be resolved, not C++ overloads");
    return CalleeCppOverloadSet{
        .cpp_overload_set_id = cpp_overload_set_type->overload_set_id,
        .self_id = fn.self_id};
  }

  if (auto impl_fn_type =
          sem_ir.insts().TryGetAs<FunctionTypeWithSelfType>(fn_type_inst_id)) {
    // Combine the associated function's `Self` with the interface function
    // data.
    fn.self_type_id = impl_fn_type->self_id;
    fn_type_inst_id = impl_fn_type->interface_function_type_id;
  }

  auto fn_type_val_id =
      sem_ir.constant_values().GetConstantInstId(fn_type_inst_id);
  if (fn_type_val_id == ErrorInst::InstId) {
    return CalleeError();
  }
  auto fn_type = sem_ir.insts().TryGetAs<FunctionType>(fn_type_val_id);
  if (!fn_type) {
    return CalleeNonFunction();
  }

  fn.function_id = fn_type->function_id;
  fn.enclosing_specific_id = fn_type->specific_id;
  return fn;
}

auto GetCalleeAsFunction(const File& sem_ir, InstId callee_id,
                         SpecificId caller_specific_id) -> CalleeFunction {
  auto callee = GetCallee(sem_ir, callee_id, caller_specific_id);
  CARBON_CHECK(std::holds_alternative<CalleeFunction>(callee));
  return std::get<CalleeFunction>(callee);
}

auto DecomposeVirtualFunction(const File& sem_ir, InstId fn_decl_id,
                              SpecificId base_class_specific_id)
    -> DecomposedVirtualFunction {
  // Remap the base's vtable entry to the appropriate constant usable in
  // the context of the derived class (for the specific for the base
  // class, for instance).
  auto fn_decl_const_id =
      GetConstantValueInSpecific(sem_ir, base_class_specific_id, fn_decl_id);
  fn_decl_id = sem_ir.constant_values().GetInstId(fn_decl_const_id);
  auto specific_id = SpecificId::None;
  auto callee_id = fn_decl_id;
  if (auto specific_function =
          sem_ir.insts().TryGetAs<SpecificFunction>(fn_decl_id)) {
    specific_id = specific_function->specific_id;
    callee_id = specific_function->callee_id;
  }

  // Identify the function we're calling by its type.
  auto fn_type_inst =
      sem_ir.types().GetAsInst(sem_ir.insts().Get(callee_id).type_id());

  return {.fn_decl_id = fn_decl_id,
          .fn_decl_const_id = fn_decl_const_id,
          .function_id = fn_type_inst.As<FunctionType>().function_id,
          .specific_id = specific_id};
}

auto Function::GetDeclaredReturnType(const File& file,
                                     SpecificId specific_id) const -> TypeId {
  if (!return_type_inst_id.has_value()) {
    return TypeId::None;
  }
  return file.types().GetTypeIdForTypeConstantId(
      GetConstantValueInSpecific(file, specific_id, return_type_inst_id));
}

auto Function::GetDeclaredReturnForm(const File& file,
                                     SpecificId specific_id) const -> InstId {
  if (!return_form_inst_id.has_value()) {
    // Treat as equivalent to `-> ()`.
    return InstId::None;
  }
  auto return_form_constant_id =
      GetConstantValueInSpecific(file, specific_id, return_form_inst_id);
  return file.constant_values().GetInstId(return_form_constant_id);
}

}  // namespace Carbon::SemIR
