// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/class.h"

#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

static auto GetFoundationType(const File& file, SpecificId specific_id,
                              InstId inst_id) -> TypeId {
  if (!inst_id.has_value()) {
    return TypeId::None;
  }
  if (inst_id == SemIR::ErrorInst::SingletonInstId) {
    return ErrorInst::SingletonTypeId;
  }
  return TypeId::ForTypeConstant(GetConstantValueInSpecific(
      file, specific_id,
      file.insts().GetAs<AnyFoundationDecl>(inst_id).foundation_type_inst_id));
}

auto Class::GetAdaptedType(const File& file, SpecificId specific_id) const
    -> TypeId {
  return GetFoundationType(file, specific_id, adapt_id);
}

auto Class::GetBaseType(const File& file, SpecificId specific_id) const
    -> TypeId {
  return GetFoundationType(file, specific_id, base_id);
}

auto Class::GetObjectRepr(const File& file, SpecificId specific_id) const
    -> TypeId {
  if (!complete_type_witness_id.has_value()) {
    return TypeId::None;
  }
  auto witness_id =
      GetConstantValueInSpecific(file, specific_id, complete_type_witness_id);
  if (witness_id == ErrorInst::SingletonConstantId) {
    return ErrorInst::SingletonTypeId;
  }
  return file.insts()
      .GetAs<CompleteTypeWitness>(file.constant_values().GetInstId(witness_id))
      .object_repr_id;
}

}  // namespace Carbon::SemIR
