// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_TYPE_H_
#define CARBON_TOOLCHAIN_SEM_IR_TYPE_H_

#include "toolchain/base/shared_value_stores.h"
#include "toolchain/sem_ir/constant.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/type_info.h"

namespace Carbon::SemIR {

// Provides a ValueStore wrapper with an API specific to types.
class TypeStore : public Yaml::Printable<TypeStore> {
 public:
  explicit TypeStore(InstStore* insts, ConstantValueStore* constants)
      : insts_(insts), constants_(constants) {}

  // Returns the ID of the constant used to define the specified type.
  auto GetConstantId(TypeId type_id) const -> ConstantId {
    if (!type_id.is_valid()) {
      // TODO: Investigate replacing this with a CHECK or returning Invalid.
      return ConstantId::NotConstant;
    }
    return type_id.AsConstantId();
  }

  // Returns the ID of the instruction used to define the specified type.
  auto GetInstId(TypeId type_id) const -> InstId {
    return constants_->GetInstId(GetConstantId(type_id));
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
    auto inst_id = constants_->GetInstId(GetConstantId(type_id));
    return insts_->GetAs<InstT>(inst_id);
  }

  // Returns the instruction used to define the specified type, if it is of a
  // particular kind.
  template <typename InstT>
  auto TryGetAs(TypeId type_id) const -> std::optional<InstT> {
    return GetAsInst(type_id).TryAs<InstT>();
  }

  // Returns whether two type IDs represent the same type. This includes the
  // case where they might be in different generics and thus might have
  // different ConstantIds, but are still symbolically equal.
  auto AreEqualAcrossDeclarations(TypeId a, TypeId b) const -> bool {
    return GetInstId(a) == GetInstId(b);
  }

  // Gets the value representation to use for a type. This returns an
  // invalid type if the given type is not complete.
  auto GetValueRepr(TypeId type_id) const -> ValueRepr {
    if (auto type_info = complete_type_info_.Lookup(type_id)) {
      return type_info.value().value_repr;
    }
    return {.kind = ValueRepr::Unknown};
  }

  // Sets the value representation associated with a type.
  auto SetValueRepr(TypeId type_id, ValueRepr value_repr) -> void {
    CARBON_CHECK(value_repr.kind != ValueRepr::Unknown);
    auto insert_info =
        complete_type_info_.Insert(type_id, {.value_repr = value_repr});
    CARBON_CHECK(insert_info.is_inserted(), "Type {0} completed more than once",
                 type_id);
    complete_types_.push_back(type_id);
    CARBON_CHECK(IsComplete(type_id));
  }

  // Determines whether the given type is known to be complete. This does not
  // determine whether the type could be completed, only whether it has been.
  auto IsComplete(TypeId type_id) const -> bool {
    return complete_type_info_.Contains(type_id);
  }

  // Determines whether the given type is a signed integer type.
  auto IsSignedInt(TypeId int_type_id) const -> bool {
    auto inst_id = GetInstId(int_type_id);
    if (inst_id == InstId::BuiltinIntLiteralType) {
      return true;
    }
    auto int_type = insts_->TryGetAs<IntType>(inst_id);
    return int_type && int_type->int_kind.is_signed();
  }

  // Returns a list of types that were completed in this file, in the order in
  // which they were completed. Earlier types in this list cannot contain
  // instances of later types.
  auto complete_types() const -> llvm::ArrayRef<TypeId> {
    return complete_types_;
  }

  auto OutputYaml() const -> Yaml::OutputMapping {
    return Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
      for (auto type_id : complete_types_) {
        map.Add(PrintToString(type_id),
                Yaml::OutputScalar(GetValueRepr(type_id)));
      }
    });
  }

  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void {
    mem_usage.Collect(MemUsage::ConcatLabel(label, "complete_type_info_"),
                      complete_type_info_);
    mem_usage.Collect(MemUsage::ConcatLabel(label, "complete_types_"),
                      complete_types_);
  }

 private:
  InstStore* insts_;
  ConstantValueStore* constants_;
  Map<TypeId, CompleteTypeInfo> complete_type_info_;
  llvm::SmallVector<TypeId> complete_types_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_TYPE_H_
