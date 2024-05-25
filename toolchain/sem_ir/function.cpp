// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/function.h"

#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

auto GetCalleeFunction(const File& sem_ir, InstId callee_id) -> CalleeFunction {
  CalleeFunction result = {.function_id = FunctionId::Invalid,
                           .self_id = InstId::Invalid,
                           .is_error = false};

  if (auto bound_method = sem_ir.insts().TryGetAs<BoundMethod>(callee_id)) {
    result.self_id = bound_method->object_id;
    callee_id = bound_method->function_id;
  }

  // Identify the function we're calling.
  auto val_id = sem_ir.constant_values().Get(callee_id);
  if (!val_id.is_constant()) {
    return result;
  }
  auto val_inst = sem_ir.insts().Get(val_id.inst_id());
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
  return result;
}

}  // namespace Carbon::SemIR
