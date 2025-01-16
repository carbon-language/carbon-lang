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

    auto operator==(const ImplsConstraint& rhs) const -> bool {
      return interface_id == rhs.interface_id && specific_id == rhs.specific_id;
    }
    // Canonically ordered by the numerical ids.
    auto operator<=>(const ImplsConstraint& rhs) const -> std::strong_ordering {
      return std::tie(interface_id.index, specific_id.index) <=>
             std::tie(rhs.interface_id.index, rhs.specific_id.index);
    }
  };
  llvm::SmallVector<ImplsConstraint> impls_constraints;

  // Rewrite constraints of the form `.T = U`
  struct RewriteConstraint {
    ConstantId lhs_const_id;
    ConstantId rhs_const_id;

    auto operator==(const RewriteConstraint& rhs) const -> bool {
      return lhs_const_id == rhs.lhs_const_id &&
             rhs_const_id == rhs.rhs_const_id;
    }
    // Canonically ordered by the numerical ids.
    auto operator<=>(const RewriteConstraint& rhs) const
        -> std::strong_ordering {
      return std::tie(lhs_const_id.index, rhs_const_id.index) <=>
             std::tie(rhs.lhs_const_id.index, rhs.rhs_const_id.index);
    }
  };
  llvm::SmallVector<RewriteConstraint> rewrite_constraints;

  // TODO: Add same-type constraints.
  // TODO: Remove once all requirements are supported.
  bool other_requirements;

  // Optional resolved facet type. For facet types used in contexts that require
  // them to be fully defined.
  ResolvedFacetTypeId resolved_id;

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

  auto operator==(const FacetTypeInfo& rhs) const -> bool {
    return impls_constraints == rhs.impls_constraints &&
           rewrite_constraints == rhs.rewrite_constraints &&
           other_requirements == rhs.other_requirements;
  }
};

struct ResolvedFacetType {
  struct RequiredInterface {
    InterfaceId interface_id;
    SpecificId specific_id;
    // One per member of the interface designated by `interface_id`.
    llvm::SmallVector<ConstantId> associated_consts;
  };

  // Interfaces mentioned explicitly in the facet type expression, or
  // transitively through a named constraint.
  llvm::SmallVector<RequiredInterface> required_interfaces;

  // Number of interfaces from `interfaces` to implement if this is the facet
  // type to the right of an `impl`...`as`. Invalid to use in that position
  // unless this value is 1.
  int num_to_impl;
};

// See common/hashing.h.
inline auto CarbonHashValue(const FacetTypeInfo& value, uint64_t seed)
    -> HashCode {
  Hasher hasher(seed);
  hasher.HashSizedBytes(llvm::ArrayRef(value.impls_constraints));
  hasher.HashSizedBytes(llvm::ArrayRef(value.rewrite_constraints));
  hasher.HashRaw(value.other_requirements);
  // `resolved_id` is not part of the state to hash.
  return static_cast<HashCode>(hasher);
}

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_FACET_TYPE_INFO_H_
