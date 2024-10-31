// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_FACET_TYPE_INFO_H_
#define CARBON_TOOLCHAIN_SEM_IR_FACET_TYPE_INFO_H_

#include "common/hashing.h"
#include "llvm/ADT/StringExtras.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

struct FacetTypeInfo : Printable<FacetTypeInfo> {
  // TODO: Need to switch to a processed, canonical form, that can support facet
  // type equality as defined by
  // https://github.com/carbon-language/carbon-lang/issues/2409.

  // The interfaces this facet type requires, sorted in numerical id order.
  llvm::SmallVector<TypeId> interface_type_ids;
  InstBlockId requirement_block_id;

  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{interfaces: ";
    llvm::ListSeparator sep;
    for (TypeId type_id : interface_type_ids) {
      out << sep << type_id;
    }
    out << "; requirements: " << requirement_block_id << "}";
  }

  auto operator==(const FacetTypeInfo& rhs) const -> bool {
    return interface_type_ids == rhs.interface_type_ids &&
           requirement_block_id == rhs.requirement_block_id;
  }
};

// See common/hashing.h.
inline auto CarbonHashValue(const FacetTypeInfo& value, uint64_t seed)
    -> HashCode {
  Hasher hasher(seed);
  hasher.HashSizedBytes(llvm::ArrayRef(value.interface_type_ids));
  hasher.HashRaw(value.requirement_block_id);
  return static_cast<HashCode>(hasher);
}

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_FACET_TYPE_INFO_H_
