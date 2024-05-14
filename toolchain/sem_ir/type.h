// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_TYPE_H_
#define CARBON_TOOLCHAIN_SEM_IR_TYPE_H_

#include "toolchain/base/value_store.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/type_info.h"

namespace Carbon::SemIR {

// Provides a ValueStore wrapper with an API specific to types.
class TypeStore : public ValueStore<TypeId> {
 public:
  explicit TypeStore(InstStore* insts) : insts_(insts) {}

  // Returns the ID of the constant used to define the specified type.
  auto GetConstantId(TypeId type_id) const -> ConstantId {
    if (type_id == TypeId::TypeType) {
      return ConstantId::ForTemplateConstant(InstId::BuiltinTypeType);
    } else if (type_id == TypeId::Error) {
      return ConstantId::Error;
    } else if (!type_id.is_valid()) {
      // TODO: Can we CHECK-fail on this?
      return ConstantId::NotConstant;
    } else {
      return Get(type_id).constant_id;
    }
  }

  // Returns the ID of the instruction used to define the specified type.
  auto GetInstId(TypeId type_id) const -> InstId {
    return GetConstantId(type_id).inst_id();
  }

  // Returns the instruction used to define the specified type.
  auto GetAsInst(TypeId type_id) const -> Inst {
    return insts_->Get(GetInstId(type_id));
  }

  // Returns whether the specified kind of instruction was used to define the
  // type.
  template <typename InstT>
  auto Is(TypeId type_id) const -> bool {
    return GetAsInst(type_id).Is<InstT>();
  }

  // Returns the instruction used to define the specified type, which is known
  // to be a particular kind of instruction.
  template <typename InstT>
  auto GetAs(TypeId type_id) const -> InstT {
    if constexpr (std::is_same_v<InstT, Builtin>) {
      return GetAsInst(type_id).As<InstT>();
    } else {
      // The type is not a builtin, so no need to check for special values.
      return insts_->Get(Get(type_id).constant_id.inst_id()).As<InstT>();
    }
  }

  // Returns the instruction used to define the specified type, if it is of a
  // particular kind.
  template <typename InstT>
  auto TryGetAs(TypeId type_id) const -> std::optional<InstT> {
    return GetAsInst(type_id).TryAs<InstT>();
  }

  // Gets the value representation to use for a type. This returns an
  // invalid type if the given type is not complete.
  auto GetValueRepr(TypeId type_id) const -> ValueRepr {
    if (type_id.index < 0) {
      // TypeType and InvalidType are their own value representation.
      return {.kind = ValueRepr::Copy, .type_id = type_id};
    }
    return Get(type_id).value_repr;
  }

  // Determines whether the given type is known to be complete. This does not
  // determine whether the type could be completed, only whether it has been.
  auto IsComplete(TypeId type_id) const -> bool {
    return GetValueRepr(type_id).kind != ValueRepr::Unknown;
  }

  // Determines whether the given type is a signed integer type.
  auto IsSignedInt(TypeId int_type_id) const -> bool {
    auto inst_id = GetInstId(int_type_id);
    if (inst_id == InstId::BuiltinIntType) {
      return true;
    }
    auto int_type = insts_->TryGetAs<IntType>(inst_id);
    return int_type && int_type->int_kind.is_signed();
  }

 private:
  InstStore* insts_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_TYPE_H_
