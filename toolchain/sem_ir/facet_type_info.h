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
  // Returns a FacetTypeInfo that combines `lhs` and `rhs`. It is not
  // canonicalized, so that it can be further modified by the caller if desired.
  static auto Combine(const FacetTypeInfo& lhs, const FacetTypeInfo& rhs)
      -> FacetTypeInfo;

  // TODO: Need to switch to a processed, canonical form, that can support facet
  // type equality as defined by
  // https://github.com/carbon-language/carbon-lang/issues/2409.

  // TODO: Replace these vectors with an array allocated in an
  // `llvm::BumpPtrAllocator`.

  // `ImplsConstraint` holds the interfaces this facet type requires.
  // TODO: extend this so it can represent named constraint requirements
  // and requirements on members, not just `.Self`.
  using ImplsConstraint = SpecificInterface;
  // These are the required interfaces that are lookup contexts.
  llvm::SmallVector<ImplsConstraint> extend_constraints;
  // These are the required interfaces that are not lookup contexts.
  llvm::SmallVector<ImplsConstraint> self_impls_constraints;

  // Rewrite constraints of the form `.T = U`
  struct RewriteConstraint {
    InstId lhs_id;
    InstId rhs_id;

    static const RewriteConstraint None;

    friend auto operator==(const RewriteConstraint& lhs,
                           const RewriteConstraint& rhs) -> bool = default;
  };
  llvm::SmallVector<RewriteConstraint> rewrite_constraints;

  // TODO: Add same-type constraints.
  // TODO: Remove once all requirements are supported.
  bool other_requirements;

  // Sorts and deduplicates constraints. Call after building the value, and then
  // don't mutate this value afterwards.
  auto Canonicalize() -> void;

  auto Print(llvm::raw_ostream& out) const -> void;

  // In some cases, a facet type is expected to represent a single interface.
  // For example, an interface declaration or an associated constant are
  // associated with a facet type that will always be a single interface with no
  // other constraints. This returns the single interface that this facet type
  // represents, or `std::nullopt` if it has any other constraints.
  auto TryAsSingleInterface() const -> std::optional<ImplsConstraint> {
    if (extend_constraints.size() == 1 && self_impls_constraints.empty() &&
        rewrite_constraints.empty() && !other_requirements) {
      return extend_constraints.front();
    }
    return std::nullopt;
  }

  friend auto operator==(const FacetTypeInfo& lhs, const FacetTypeInfo& rhs)
      -> bool {
    return lhs.extend_constraints == rhs.extend_constraints &&
           lhs.self_impls_constraints == rhs.self_impls_constraints &&
           lhs.rewrite_constraints == rhs.rewrite_constraints &&
           lhs.other_requirements == rhs.other_requirements;
  }
};

constexpr FacetTypeInfo::RewriteConstraint
    FacetTypeInfo::RewriteConstraint::None = {.lhs_id = InstId::None,
                                              .rhs_id = InstId::None};

struct IdentifiedFacetType {
  using RequiredInterface = SpecificInterface;

  IdentifiedFacetType(llvm::ArrayRef<RequiredInterface> extends,
                      llvm::ArrayRef<RequiredInterface> self_impls);

  // The order here defines the order of impl witnesses for this facet type.
  auto required_interfaces() const -> llvm::ArrayRef<RequiredInterface> {
    return required_interfaces_;
  }

  // Can this be used to the right of an `as` in an `impl` declaration?
  auto is_valid_impl_as_target() const -> bool {
    return interface_id_.has_value();
  }

  // The interface to implement when this facet type is used in an `impl`
  // declaration.
  auto impl_as_target_interface() const -> SpecificInterface {
    if (is_valid_impl_as_target()) {
      return {.interface_id = interface_id_, .specific_id = specific_id_};
    } else {
      return SpecificInterface::None;
    }
  }

  auto num_interfaces_to_impl() const -> int {
    if (is_valid_impl_as_target()) {
      return 1;
    } else {
      return num_interface_to_impl_;
    }
  }

 private:
  // Interfaces mentioned explicitly in the facet type expression, or
  // transitively through a named constraint. Sorted and deduplicated.
  llvm::SmallVector<RequiredInterface> required_interfaces_;

  // The single interface from `required_interfaces` to implement if this is
  // the facet type to the right of an `impl`...`as`, or `None` if no such
  // single interface.
  InterfaceId interface_id_ = InterfaceId::None;
  union {
    // If `interface_id` is `None`, the number of interfaces to report in a
    // diagnostic about why this facet type can't be implemented.
    int num_interface_to_impl_ = 0;
    // If `interface_id` is not `None`, the specific for that interface.
    SpecificId specific_id_;
  };
};

// See common/hashing.h.
inline auto CarbonHashValue(const FacetTypeInfo& value, uint64_t seed)
    -> HashCode {
  Hasher hasher(seed);
  hasher.HashSizedBytes(llvm::ArrayRef(value.extend_constraints));
  hasher.HashSizedBytes(llvm::ArrayRef(value.self_impls_constraints));
  hasher.HashSizedBytes(llvm::ArrayRef(value.rewrite_constraints));
  hasher.HashRaw(value.other_requirements);
  // `complete_id` is not part of the state to hash.
  return static_cast<HashCode>(hasher);
}

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_FACET_TYPE_INFO_H_
