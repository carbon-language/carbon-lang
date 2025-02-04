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

  // `ImplsConstraint` holds the interfaces this facet type requires.
  struct ImplsConstraint {
    // TODO: extend this so it can represent named constraint requirements
    // and requirements on members, not just `.Self`.
    // TODO: Add whether this is a lookup context. Those that are should sort
    // first for easy access. Right now, all are assumed to be lookup contexts.
    InterfaceId interface_id;
    SpecificId specific_id;

    friend auto operator==(const ImplsConstraint& lhs,
                           const ImplsConstraint& rhs) -> bool {
      return lhs.interface_id == rhs.interface_id &&
             lhs.specific_id == rhs.specific_id;
    }
    // Canonically ordered by the numerical ids.
    friend auto operator<=>(const ImplsConstraint& lhs,
                            const ImplsConstraint& rhs)
        -> std::strong_ordering {
      return std::tie(lhs.interface_id.index, lhs.specific_id.index) <=>
             std::tie(rhs.interface_id.index, rhs.specific_id.index);
    }
  };
  llvm::SmallVector<ImplsConstraint> impls_constraints;

  // Rewrite constraints of the form `.T = U`
  struct RewriteConstraint {
    ConstantId lhs_const_id;
    ConstantId rhs_const_id;

    friend auto operator==(const RewriteConstraint& lhs,
                           const RewriteConstraint& rhs) -> bool {
      return lhs.lhs_const_id == rhs.lhs_const_id &&
             lhs.rhs_const_id == rhs.rhs_const_id;
    }
    // Canonically ordered by the numerical ids.
    friend auto operator<=>(const RewriteConstraint& lhs,
                            const RewriteConstraint& rhs)
        -> std::strong_ordering {
      return std::tie(lhs.lhs_const_id.index, lhs.rhs_const_id.index) <=>
             std::tie(rhs.lhs_const_id.index, rhs.rhs_const_id.index);
    }
  };
  llvm::SmallVector<RewriteConstraint> rewrite_constraints;

  // TODO: Add same-type constraints.
  // TODO: Remove once all requirements are supported.
  bool other_requirements;
  // TODO: Add optional resolved facet type.

  // Sorts and deduplicates constraints.
  auto Canonicalize() -> void;

  auto Print(llvm::raw_ostream& out) const -> void;

  // TODO: Update callers to be able to deal with facet types that aren't a
  // single interface and then remove this function.
  auto TryAsSingleInterface() const -> std::optional<ImplsConstraint> {
    // We are ignoring other requirements for the moment, since this function is
    // (hopefully) temporary.
    if (impls_constraints.size() == 1) {
      return impls_constraints.front();
    }
    return std::nullopt;
  }

  friend auto operator==(const FacetTypeInfo& lhs, const FacetTypeInfo& rhs)
      -> bool {
    return lhs.impls_constraints == rhs.impls_constraints &&
           lhs.rewrite_constraints == rhs.rewrite_constraints &&
           lhs.other_requirements == rhs.other_requirements;
  }
};

// See common/hashing.h.
inline auto CarbonHashValue(const FacetTypeInfo& value, uint64_t seed)
    -> HashCode {
  Hasher hasher(seed);
  hasher.HashSizedBytes(llvm::ArrayRef(value.impls_constraints));
  hasher.HashSizedBytes(llvm::ArrayRef(value.rewrite_constraints));
  hasher.HashRaw(value.other_requirements);
  return static_cast<HashCode>(hasher);
}

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_FACET_TYPE_INFO_H_
