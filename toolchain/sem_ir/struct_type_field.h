// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_STRUCT_TYPE_FIELD_H_
#define CARBON_TOOLCHAIN_SEM_IR_STRUCT_TYPE_FIELD_H_

#include "toolchain/sem_ir/block_value_store.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// A field in a struct's type, such as `.a: i32` in `{.a: i32}`.
struct StructTypeField : Printable<StructTypeField> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{name_id: " << name_id << ", type_id: " << type_id << "}";
  }

  friend auto operator==(StructTypeField lhs, StructTypeField rhs) -> bool {
    return std::memcmp(&lhs, &rhs, sizeof(StructTypeField)) == 0;
  }

  NameId name_id;
  TypeId type_id;
};

using StructTypeFieldsStore = BlockValueStore<StructTypeFieldsId>;

// See common/hashing.h. Supports canonicalization of fields.
inline auto CarbonHashValue(const StructTypeField& value, uint64_t seed)
    -> HashCode {
  Hasher hasher(seed);
  hasher.HashRaw(value);
  return static_cast<HashCode>(hasher);
}

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_STRUCT_TYPE_FIELD_H_
