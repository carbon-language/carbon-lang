// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/type.h"

#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

auto TypeStore::GetInstId(TypeId type_id) const -> InstId {
  return file_->constant_values().GetInstId(GetConstantId(type_id));
}

auto TypeStore::GetAsInst(TypeId type_id) const -> Inst {
  return file_->insts().Get(GetInstId(type_id));
}

auto TypeStore::GetObjectRepr(TypeId type_id) const -> TypeId {
  type_id = GetUnqualifiedType(type_id);
  auto class_type = TryGetAs<ClassType>(type_id);
  if (!class_type) {
    return type_id;
  }
  const auto& class_info = file_->classes().Get(class_type->class_id);
  if (!class_info.is_defined()) {
    return TypeId::Invalid;
  }
  return class_info.GetObjectRepr(*file_, class_type->specific_id);
}

auto TypeStore::GetUnqualifiedType(TypeId type_id) const -> TypeId {
  if (auto const_type = TryGetAs<ConstType>(type_id)) {
    return const_type->inner_id;
  }
  return type_id;
}

auto TypeStore::IsSignedInt(TypeId int_type_id) const -> bool {
  auto object_repr_id = GetObjectRepr(int_type_id);
  if (!object_repr_id.is_valid()) {
    return false;
  }
  auto inst_id = GetInstId(int_type_id);
  if (inst_id == InstId::BuiltinIntLiteralType) {
    return true;
  }
  auto int_type = file_->insts().TryGetAs<IntType>(inst_id);
  return int_type && int_type->int_kind.is_signed();
}

auto TypeStore::GetIntTypeInfo(TypeId int_type_id) const -> IntTypeInfo {
  auto object_repr_id = GetObjectRepr(int_type_id);
  auto inst_id = GetInstId(object_repr_id);
  if (inst_id == InstId::BuiltinIntLiteralType) {
    return {.is_signed = true, .bit_width = IntId::Invalid};
  }
  auto int_type = file_->insts().GetAs<IntType>(inst_id);
  auto bit_width_inst =
      file_->insts().TryGetAs<IntValue>(int_type.bit_width_id);
  return {
      .is_signed = int_type.int_kind.is_signed(),
      .bit_width = bit_width_inst ? bit_width_inst->int_id : IntId::Invalid};
}

}  // namespace Carbon::SemIR
