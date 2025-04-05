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
      -> FacetTypeInfo {
    FacetTypeInfo info = {.other_requirements = false};
    llvm::append_range(info.impls_constraints, lhs.impls_constraints);
    llvm::append_range(info.impls_constraints, rhs.impls_constraints);
    llvm::append_range(info.rewrite_constraints, lhs.rewrite_constraints);
    llvm::append_range(info.rewrite_constraints, rhs.rewrite_constraints);
    info.other_requirements |= lhs.other_requirements;
    info.other_requirements |= rhs.other_requirements;
    return info;
  }

  // TODO: Need to switch to a processed, canonical form, that can support facet
  // type equality as defined by
  // https://github.com/carbon-language/carbon-lang/issues/2409.

  // TODO: Replace these vectors with an array allocated in an
  // `llvm::BumpPtrAllocator`.

  // `ImplsConstraint` holds the interfaces this facet type requires.
  // TODO: extend this so it can represent named constraint requirements
  // and requirements on members, not just `.Self`.
  // TODO: Add whether this is a lookup context. Those that are should sort
  // first for easy access. Right now, all are assumed to be lookup contexts.
  using ImplsConstraint = SpecificInterface;
  llvm::SmallVector<ImplsConstraint> impls_constraints;

  // Rewrite constraints of the form `.T = U`
  struct RewriteConstraint {
    ConstantId lhs_const_id;
    ConstantId rhs_const_id;

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
  // associated with a facet type that will always be a single interface.
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

constexpr FacetTypeInfo::RewriteConstraint
    FacetTypeInfo::RewriteConstraint::None = {.lhs_const_id = ConstantId::None,
                                              .rhs_const_id = ConstantId::None};

struct IdentifiedFacetType {
  using RequiredInterface = SpecificInterface;

  IdentifiedFacetType() {}

  auto required_interfaces() const -> llvm::ArrayRef<RequiredInterface> {
    return required_interfaces_;
  }

  auto set_required_interfaces(const llvm::ArrayRef<RequiredInterface> set_to) {
    required_interfaces_.assign(set_to.begin(), set_to.end());
    CanonicalizeRequiredInterfaces();
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

  // Call this function if num != 1, otherwise call `set_interface_to_impl`.
  auto set_num_interfaces_to_impl(int num) -> void {
    CARBON_CHECK(num != 1);
    interface_id_ = InterfaceId::None;
    num_interface_to_impl_ = num;
  }

  // If there is a single interface to implement, specify which it is.
  // Should be an element of `required_interfaces()`.
  auto set_interface_to_impl(SpecificInterface interface) -> void {
    CARBON_CHECK(interface.interface_id.has_value());
    interface_id_ = interface.interface_id;
    specific_id_ = interface.specific_id;
  }

 private:
  // Sorts and deduplicates `required_interfaces`. Call after building the sets
  // of interfaces, and then don't mutate the value afterwards.
  auto CanonicalizeRequiredInterfaces() -> void;

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
  hasher.HashSizedBytes(llvm::ArrayRef(value.impls_constraints));
  hasher.HashSizedBytes(llvm::ArrayRef(value.rewrite_constraints));
  hasher.HashRaw(value.other_requirements);
  // `complete_id` is not part of the state to hash.
  return static_cast<HashCode>(hasher);
}

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_FACET_TYPE_INFO_H_
