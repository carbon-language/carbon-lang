// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/type.h"

#include <optional>

#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

CARBON_DEFINE_ENUM_MASK_NAMES(TypeQualifiers) = {
    CARBON_TYPE_QUALIFIERS(CARBON_ENUM_MASK_NAME_STRING)};

// Verify that the constant value's type is `TypeType` (or an error).
static void CheckTypeOfConstantIsTypeType(File& file, ConstantId constant_id) {
  CARBON_CHECK(constant_id.is_constant(),
               "Canonicalizing non-constant type: {0}", constant_id);
  auto type_id =
      file.insts().Get(file.constant_values().GetInstId(constant_id)).type_id();
  CARBON_CHECK(
      type_id == TypeType::TypeId || constant_id == ErrorInst::ConstantId,
      "Forming type ID for non-type constant of type {0}",
      file.types().GetAsInst(type_id));
}

auto TypeStore::GetTypeIdForTypeConstantId(ConstantId constant_id) const
    -> TypeId {
  CheckTypeOfConstantIsTypeType(*file_, constant_id);
  return TypeId::ForTypeConstant(constant_id);
}

auto TypeStore::GetTypeIdForTypeInstId(InstId inst_id) const -> TypeId {
  auto constant_id = file_->constant_values().Get(inst_id);
  CheckTypeOfConstantIsTypeType(*file_, constant_id);
  return TypeId::ForTypeConstant(constant_id);
}

auto TypeStore::GetTypeIdForTypeInstId(TypeInstId inst_id) const -> TypeId {
  auto constant_id = file_->constant_values().Get(inst_id);
  return TypeId::ForTypeConstant(constant_id);
}

auto TypeStore::GetAsTypeInstId(InstId inst_id) const -> TypeInstId {
  auto constant_id = file_->constant_values().Get(inst_id);
  CheckTypeOfConstantIsTypeType(*file_, constant_id);
  return TypeInstId::UnsafeMake(inst_id);
}

auto TypeStore::GetInstId(TypeId type_id) const -> TypeInstId {
  // The instruction for a TypeId has a value of that TypeId.
  return TypeInstId::UnsafeMake(
      file_->constant_values().GetInstId(GetConstantId(type_id)));
}

auto TypeStore::GetAsInst(TypeId type_id) const -> Inst {
  return file_->insts().Get(GetInstId(type_id));
}

auto TypeStore::GetUnattachedType(TypeId type_id) const -> TypeId {
  return TypeId::ForTypeConstant(
      file_->constant_values().GetUnattachedConstant(type_id.AsConstantId()));
}

auto TypeStore::GetObjectRepr(TypeId type_id) const -> TypeId {
  type_id = GetUnqualifiedType(type_id);
  auto class_type = TryGetAs<ClassType>(type_id);
  if (!class_type) {
    return type_id;
  }
  const auto& class_info = file_->classes().Get(class_type->class_id);
  if (!class_info.is_complete()) {
    return TypeId::None;
  }
  return class_info.GetObjectRepr(*file_, class_type->specific_id);
}

auto TypeStore::GetAdaptedType(TypeId type_id) const -> TypeId {
  if (auto class_type = TryGetAs<ClassType>(type_id)) {
    return file_->classes()
        .Get(class_type->class_id)
        .GetAdaptedType(*file_, class_type->specific_id);
  }
  return TypeId::None;
}

auto TypeStore::GetTransitiveAdaptedType(TypeId type_id) const -> TypeId {
  while (true) {
    auto adapted_type_id = GetAdaptedType(type_id);
    if (!adapted_type_id.has_value()) {
      break;
    }
    type_id = adapted_type_id;
  }
  return type_id;
}

auto TypeStore::GetUnqualifiedTypeAndQualifiers(TypeId type_id) const
    -> std::pair<TypeId, TypeQualifiers> {
  TypeQualifiers quals = TypeQualifiers::None;
  while (true) {
    if (auto qualified_type = TryGetAs<AnyQualifiedType>(type_id)) {
      type_id = file_->types().GetTypeIdForTypeInstId(qualified_type->inner_id);
      switch (qualified_type->kind) {
        case ConstType::Kind:
          quals.Add(TypeQualifiers::Const);
          break;
        case MaybeUnformedType::Kind:
          quals.Add(TypeQualifiers::MaybeUnformed);
          break;
        case PartialType::Kind:
          quals.Add(TypeQualifiers::Partial);
          break;
        default:
          CARBON_FATAL("Unknown type qualifier {0}", qualified_type->kind);
      }
    } else {
      return {type_id, quals};
    }
  }
}

auto TypeStore::GetTransitiveUnqualifiedAdaptedType(TypeId type_id) const
    -> std::pair<TypeId, TypeQualifiers> {
  TypeQualifiers quals = TypeQualifiers::None;
  while (true) {
    type_id = GetTransitiveAdaptedType(type_id);
    auto [unqual_type_id, inner_quals] =
        GetUnqualifiedTypeAndQualifiers(type_id);
    if (unqual_type_id == type_id) {
      return {type_id, quals};
    }
    type_id = unqual_type_id;
    quals.Add(inner_quals);
  }
}

auto TypeStore::TryGetIntTypeInfo(TypeId int_type_id) const
    -> std::optional<IntTypeInfo> {
  auto object_repr_id = file_->types().GetObjectRepr(int_type_id);
  if (!object_repr_id.has_value()) {
    return std::nullopt;
  }
  auto inst_id = file_->types().GetInstId(object_repr_id);
  if (inst_id == IntLiteralType::TypeInstId) {
    // `Core.IntLiteral` has an unknown bit-width.
    return TypeStore::IntTypeInfo{.is_signed = true, .bit_width = IntId::None};
  }
  auto int_type = file_->insts().TryGetAs<IntType>(inst_id);
  if (!int_type) {
    return std::nullopt;
  }
  auto bit_width_inst =
      file_->insts().TryGetAs<IntValue>(int_type->bit_width_id);
  return TypeStore::IntTypeInfo{
      .is_signed = int_type->int_kind.is_signed(),
      .bit_width = bit_width_inst ? bit_width_inst->int_id : IntId::None};
}

auto TypeStore::IsSignedInt(TypeId int_type_id) const -> bool {
  auto int_info = TryGetIntTypeInfo(int_type_id);
  return int_info && int_info->is_signed;
}

auto TypeStore::GetIntTypeInfo(TypeId int_type_id) const -> IntTypeInfo {
  auto int_info = TryGetIntTypeInfo(int_type_id);
  CARBON_CHECK(int_info, "Type {0} is not an integer type", int_type_id);
  return *int_info;
}

auto ExtractScrutineeType(const File& sem_ir, SemIR::TypeId type_id)
    -> SemIR::TypeId {
  if (auto pattern_type =
          sem_ir.types().TryGetAs<SemIR::PatternType>(type_id)) {
    return sem_ir.types().GetTypeIdForTypeInstId(
        pattern_type->scrutinee_type_inst_id);
  }
  CARBON_CHECK(type_id == SemIR::ErrorInst::TypeId,
               "Inst kind doesn't have scrutinee type: {0}",
               sem_ir.types().GetAsInst(type_id).kind());
  return type_id;
}

}  // namespace Carbon::SemIR
