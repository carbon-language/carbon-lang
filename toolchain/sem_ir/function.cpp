// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/function.h"

#include <optional>

#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

auto GetCalleeFunction(const File& sem_ir, InstId callee_id,
                       SpecificId specific_id) -> CalleeFunction {
  CalleeFunction result = {.function_id = FunctionId::None,
                           .enclosing_specific_id = SpecificId::None,
                           .resolved_specific_id = SpecificId::None,
                           .self_type_id = InstId::None,
                           .self_id = InstId::None,
                           .is_error = false,
                           .is_cpp_overload_set = false};
  if (auto bound_method = sem_ir.insts().TryGetAs<BoundMethod>(callee_id)) {
    result.self_id = bound_method->object_id;
    callee_id = bound_method->function_decl_id;
  }

  if (specific_id.has_value()) {
    callee_id = sem_ir.constant_values().GetInstIdIfValid(
        GetConstantValueInSpecific(sem_ir, specific_id, callee_id));
    CARBON_CHECK(callee_id.has_value(),
                 "Invalid callee id in a specific context");
  }

  if (auto specific_function =
          sem_ir.insts().TryGetAs<SpecificFunction>(callee_id)) {
    result.resolved_specific_id = specific_function->specific_id;
    callee_id = specific_function->callee_id;
  }

  // Identify the function we're calling by its type.
  auto val_id = sem_ir.constant_values().GetConstantInstId(callee_id);
  if (!val_id.has_value()) {
    return result;
  }
  auto fn_type_inst =
      sem_ir.types().GetAsInst(sem_ir.insts().Get(val_id).type_id());

  if (fn_type_inst.TryAs<CppOverloadSetType>()) {
    // TODO: Consider evaluating this at runtime instead of having a field.
    result.is_cpp_overload_set = true;
    return result;
  }

  if (auto impl_fn_type = fn_type_inst.TryAs<FunctionTypeWithSelfType>()) {
    // Combine the associated function's `Self` with the interface function
    // data.
    result.self_type_id = impl_fn_type->self_id;
    fn_type_inst = sem_ir.insts().Get(impl_fn_type->interface_function_type_id);
  }

  auto fn_type = fn_type_inst.TryAs<FunctionType>();
  if (!fn_type) {
    result.is_error = fn_type_inst.Is<ErrorInst>();
    return result;
  }

  result.function_id = fn_type->function_id;
  result.enclosing_specific_id = fn_type->specific_id;
  return result;
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
  auto specific_id = SemIR::SpecificId::None;
  auto callee_id = fn_decl_id;
  if (auto specific_function =
          sem_ir.insts().TryGetAs<SemIR::SpecificFunction>(fn_decl_id)) {
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

auto Function::GetParamPatternInfoFromPatternId(const File& sem_ir,
                                                InstId pattern_id)
    -> std::optional<ParamPatternInfo> {
  auto inst_id = pattern_id;
  auto inst = sem_ir.insts().Get(inst_id);

  sem_ir.insts().TryUnwrap(inst, inst_id, &AddrPattern::inner_id);
  auto [var_pattern, var_pattern_id] =
      sem_ir.insts().TryUnwrap(inst, inst_id, &VarPattern::subpattern_id);
  auto [param_pattern, param_pattern_id] =
      sem_ir.insts().TryUnwrap(inst, inst_id, &AnyParamPattern::subpattern_id);
  if (!param_pattern) {
    return std::nullopt;
  }

  auto binding_pattern = inst.As<AnyBindingPattern>();
  return {{.inst_id = param_pattern_id,
           .inst = *param_pattern,
           .entity_name_id = binding_pattern.entity_name_id,
           .var_pattern_id = var_pattern_id}};
}

auto Function::GetDeclaredReturnType(const File& file,
                                     SpecificId specific_id) const -> TypeId {
  if (!return_slot_pattern_id.has_value()) {
    return TypeId::None;
  }
  return ExtractScrutineeType(
      file, GetTypeOfInstInSpecific(file, specific_id, return_slot_pattern_id));
}

}  // namespace Carbon::SemIR
