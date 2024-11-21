// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/class.h"

#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

auto Class::GetAdaptedType(const File& file, SpecificId specific_id) const
    -> TypeId {
  if (!adapt_id.is_valid()) {
    return TypeId::Invalid;
  }
  if (base_id == SemIR::InstId::BuiltinErrorInst) {
    return TypeId::Error;
  }
  return TypeId::ForTypeConstant(GetConstantValueInSpecific(
      file, specific_id,
      file.insts().GetAs<AdaptDecl>(adapt_id).adapted_type_inst_id));
}

auto Class::GetBaseType(const File& file, SpecificId specific_id) const
    -> TypeId {
  if (!base_id.is_valid()) {
    return TypeId::Invalid;
  }
  if (base_id == SemIR::InstId::BuiltinErrorInst) {
    return TypeId::Error;
  }
  CARBON_CHECK(base_id.index >= 0);
  return TypeId::ForTypeConstant(GetConstantValueInSpecific(
      file, specific_id,
      file.insts().GetAs<BaseDecl>(base_id).base_type_inst_id));
}

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
