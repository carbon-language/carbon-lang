// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/function.h"

#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

auto GetCalleeFunction(const File& sem_ir, InstId callee_id) -> CalleeFunction {
  CalleeFunction result = {.function_id = FunctionId::Invalid,
                           .enclosing_specific_id = SpecificId::Invalid,
                           .resolved_specific_id = SpecificId::Invalid,
                           .self_id = InstId::Invalid,
                           .is_error = false};

  if (auto specific_function =
          sem_ir.insts().TryGetAs<SpecificFunction>(callee_id)) {
    result.resolved_specific_id = specific_function->specific_id;
    callee_id = specific_function->callee_id;
  }

  if (auto bound_method = sem_ir.insts().TryGetAs<BoundMethod>(callee_id)) {
    result.self_id = bound_method->object_id;
    callee_id = bound_method->function_id;
  }

  // Identify the function we're calling.
  auto val_id = sem_ir.constant_values().GetConstantInstId(callee_id);
  if (!val_id.is_valid()) {
    return result;
  }
  auto val_inst = sem_ir.insts().Get(val_id);
  auto struct_val = val_inst.TryAs<StructValue>();
  if (!struct_val) {
    result.is_error = val_inst.type_id() == SemIR::TypeId::Error;
    return result;
  }
  auto fn_type = sem_ir.types().TryGetAs<FunctionType>(struct_val->type_id);
  if (!fn_type) {
    return result;
  }

  result.function_id = fn_type->function_id;
  result.enclosing_specific_id = fn_type->specific_id;
  return result;
}

auto Function::ParamPatternInfo::GetNameId(const File& sem_ir) -> NameId {
  return sem_ir.entity_names().Get(entity_name_id).name_id;
}

auto Function::GetParamPatternInfoFromPatternId(const File& sem_ir,
                                                InstId pattern_id)
    -> ParamPatternInfo {
  auto inst_id = pattern_id;
  auto inst = sem_ir.insts().Get(inst_id);

  if (auto addr_pattern = inst.TryAs<SemIR::AddrPattern>()) {
    inst_id = addr_pattern->inner_id;
    inst = sem_ir.insts().Get(inst_id);
  }

  auto param_pattern_id = inst_id;
  auto param_pattern_inst = inst.As<SemIR::AnyParamPattern>();

  inst_id = param_pattern_inst.subpattern_id;
  inst = sem_ir.insts().Get(inst_id);

  auto binding_pattern = inst.As<AnyBindingPattern>();
  return {.inst_id = param_pattern_id,
          .inst = param_pattern_inst,
          .entity_name_id = binding_pattern.entity_name_id};
}

auto Function::GetNameFromPatternId(const File& sem_ir, InstId pattern_id)
    -> SemIR::NameId {
  auto inst_id = pattern_id;
  auto inst = sem_ir.insts().Get(inst_id);

  if (auto addr_pattern = inst.TryAs<SemIR::AddrPattern>()) {
    inst_id = addr_pattern->inner_id;
    inst = sem_ir.insts().Get(inst_id);
  }

  if (inst_id == SemIR::InstId::BuiltinErrorInst) {
    return SemIR::NameId::Invalid;
  }

  auto param_pattern_inst = inst.As<SemIR::AnyParamPattern>();

  inst_id = param_pattern_inst.subpattern_id;
  inst = sem_ir.insts().Get(inst_id);

  if (inst.Is<ReturnSlotPattern>()) {
    return SemIR::NameId::ReturnSlot;
  }
  auto binding_pattern = inst.As<AnyBindingPattern>();
  return sem_ir.entity_names().Get(binding_pattern.entity_name_id).name_id;
}

auto Function::GetDeclaredReturnType(const File& file,
                                     SpecificId specific_id) const -> TypeId {
  if (!return_slot_pattern_id.is_valid()) {
    return TypeId::Invalid;
  }
  return GetTypeInSpecific(file, specific_id,
                           file.insts().Get(return_slot_pattern_id).type_id());
}

}  // namespace Carbon::SemIR
