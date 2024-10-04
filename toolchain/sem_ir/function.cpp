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
                           .specific_id = SpecificId::Invalid,
                           .self_id = InstId::Invalid,
                           .is_error = false};

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
  result.specific_id = fn_type->specific_id;
  return result;
}

auto Function::GetParamFromParamRefId(const File& sem_ir, InstId param_ref_id)
    -> ParamInfo {
  auto ref = sem_ir.insts().Get(param_ref_id);

  if (auto addr_pattern = ref.TryAs<AddrPattern>()) {
    param_ref_id = addr_pattern->inner_id;
    ref = sem_ir.insts().Get(param_ref_id);
  }

  auto bind_name = ref.TryAs<AnyBindName>();
  if (bind_name) {
    param_ref_id = bind_name->value_id;
    ref = sem_ir.insts().Get(param_ref_id);
  }
  return {param_ref_id, ref.As<Param>(), bind_name};
}

auto Function::ParamInfo::GetNameId(const File& sem_ir) -> NameId {
  if (bind_name) {
    return sem_ir.entity_names().Get(bind_name->entity_name_id).name_id;
  } else {
    return NameId::Invalid;
  }
}

auto Function::GetDeclaredReturnType(const File& file,
                                     SpecificId specific_id) const -> TypeId {
  if (!return_storage_id.is_valid()) {
    return TypeId::Invalid;
  }
  return GetTypeInSpecific(file, specific_id,
                           file.insts().Get(return_storage_id).type_id());
}

}  // namespace Carbon::SemIR
