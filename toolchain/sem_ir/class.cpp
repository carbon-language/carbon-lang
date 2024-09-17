// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/class.h"

#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

auto Class::GetObjectRepr(const File& file, SpecificId specific_id) const
    -> TypeId {
  if (!complete_type_witness_id.is_valid()) {
    return TypeId::Invalid;
  }
  auto witness_id =
      GetConstantValueInSpecific(file, specific_id, complete_type_witness_id);
  if (witness_id == ConstantId::Error) {
    return TypeId::Error;
  }
  return file.insts()
      .GetAs<CompleteTypeWitness>(file.constant_values().GetInstId(witness_id))
      .object_repr_id;
}

}  // namespace Carbon::SemIR
