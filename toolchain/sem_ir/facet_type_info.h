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

  // TODO: Replace these vectors with an array allocated in an
  // `llvm::BumpPtrAllocator`.

  // `Impls` holds the interfaces this facet type requires.
  struct Impls {
    auto operator==(const Impls& rhs) const -> bool {
      return interface_id == rhs.interface_id && specific_id == rhs.specific_id;
    }
    // Canonically ordered by the numerical ids.
    auto operator<=>(const Impls& rhs) const -> std::strong_ordering {
      return std::tie(interface_id.index, specific_id.index) <=>
             std::tie(rhs.interface_id.index, rhs.specific_id.index);
    }

    // TODO: extend this so it can represent named constraint requirements
    // and requirements on members, not just `.Self`.
    InterfaceId interface_id;
    SpecificId specific_id;
  };
  llvm::SmallVector<Impls> impls;
  // TODO: Add lookup contexts.
  // TODO: Add rewrite constraints.
  // TODO: Add same-type constraints.
  // TODO: Remove `requirement_block_id`.
  InstBlockId requirement_block_id;
  // TODO: Add optional resolved facet type.

  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{impls interface: ";
    llvm::ListSeparator sep;
    for (Impls req : impls) {
      out << sep << req.interface_id;
      if (req.specific_id.is_valid()) {
        out << "(" << req.specific_id << ")";
      }
    }
    out << "; requirements: " << requirement_block_id << "}";
  }

  // TODO: Update callers to be able to deal with facet types that aren't a
  // single interface and then remove this function.
  auto TryAsSingleInterface() const -> std::optional<Impls> {
    // We are ignoring requirement_block_id for the moment since nothing uses it
    // yet.
    if (impls.size() == 1) {
      return impls.front();
    }
    return std::nullopt;
  }

  auto operator==(const FacetTypeInfo& rhs) const -> bool {
    return impls == rhs.impls &&
           requirement_block_id == rhs.requirement_block_id;
  }
};

// See common/hashing.h.
inline auto CarbonHashValue(const FacetTypeInfo& value, uint64_t seed)
    -> HashCode {
  Hasher hasher(seed);
  hasher.HashSizedBytes(llvm::ArrayRef(value.impls));
  hasher.HashRaw(value.requirement_block_id);
  return static_cast<HashCode>(hasher);
}

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_FACET_TYPE_INFO_H_
