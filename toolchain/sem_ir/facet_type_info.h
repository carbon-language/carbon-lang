// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_FACET_TYPE_INFO_H_
#define CARBON_TOOLCHAIN_SEM_IR_FACET_TYPE_INFO_H_

#include "common/enum_mask_base.h"
#include "common/hashing.h"
#include "llvm/ADT/StringExtras.h"
#include "toolchain/base/canonical_value_store.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/specific_interface.h"
#include "toolchain/sem_ir/specific_named_constraint.h"

namespace Carbon::SemIR {

class File;

// A representation of a facet type that extends a single interface or
// named constraint.
using SingleExtendFacetType =
    std::variant<SpecificInterface, SpecificNamedConstraint>;

// The canonical description of a FacetType. Contains the interfaces, named
// constraints, and any constraints on types that are part of the facet type.
// All values within are canonical in order for comparison to be used for
// type equality.
//
// The structure keeps separate dependencies on interfaces and named
// constraints, even though named constraints ultimately just name interfaces,
// as it provides a canonical but otherwise unprocessed representation of the
// facet type.
//
// The flattening of the named constraints into interfaces is done by forming
// the IdentifiedFacetType for a specific Self type.
//
// TODO: Rename to DeclaredFacetType.
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

  // These are the required interfaces that are lookup contexts.
  llvm::SmallVector<SpecificInterface> extend_constraints;
  // These are the required interfaces that are not lookup contexts.
  llvm::SmallVector<SpecificInterface> self_impls_constraints;

  // These name constraints add interfaces as lookup contexts, if they are
  // extended in the named constraint.
  llvm::SmallVector<SpecificNamedConstraint> extend_named_constraints;
  // These name constraints don't add interfaces as lookup contexts.
  llvm::SmallVector<SpecificNamedConstraint> self_impls_named_constraints;

  // Requirements on types other than the generic self.
  struct TypeImplsInterface {
    // A facet or type value, which is required to implement the interface.
    // Must be a canonical instruction to ensure comparison works correctly.
    InstId self_type;
    SpecificInterface specific_interface;

    friend auto operator==(const TypeImplsInterface& lhs,
                           const TypeImplsInterface& rhs) -> bool = default;
  };
  struct TypeImplsNamedConstraint {
    // A facet or type value, which is required to implement the constraint.
    // Must be a canonical instruction to ensure comparison works correctly.
    InstId self_type;
    SpecificNamedConstraint specific_named_constraint;

    friend auto operator==(const TypeImplsNamedConstraint& lhs,
                           const TypeImplsNamedConstraint& rhs)
        -> bool = default;
  };
  llvm::SmallVector<TypeImplsInterface> type_impls_interfaces;
  llvm::SmallVector<TypeImplsNamedConstraint> type_impls_named_constraints;

  // Rewrite constraints of the form `.T = U`.
  //
  // The InstIds here must be canonical instructions (which come from the
  // instruction in a constant value) in order to ensure comparison works
  // correctly.
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
  bool other_requirements = false;

  // Sorts and deduplicates constraints. Call after building the value, and then
  // don't mutate this value afterwards.
  auto Canonicalize() -> void;

  auto Print(llvm::raw_ostream& out) const -> void;

  // In some cases, a facet type is expected to represent a single interface or
  // named constraint. For example, an interface declaration, or an associated
  // constant are associated with a facet type that will always be a single
  // interface with no other requirements. This returns the single interface or
  // named constraint that this facet type represents, or `std::nullopt` if it
  // has any other requirements.
  auto TryAsSingleExtend() const -> std::optional<SingleExtendFacetType> {
    if (!self_impls_constraints.empty() ||
        !self_impls_named_constraints.empty() ||
        !type_impls_interfaces.empty() ||
        !type_impls_named_constraints.empty() || !rewrite_constraints.empty() ||
        other_requirements) {
      return std::nullopt;
    }
    if (extend_constraints.size() == 1 && extend_named_constraints.empty()) {
      return extend_constraints.front();
    }
    if (extend_constraints.empty() && extend_named_constraints.size() == 1) {
      return extend_named_constraints.front();
    }
    return std::nullopt;
  }

  // Returns whether the facet type has no constraints, making it the facet type
  // version of `TypeType`.
  auto HasNoConstraints() const -> bool {
    return extend_constraints.empty() && extend_named_constraints.empty() &&
           self_impls_constraints.empty() &&
           self_impls_named_constraints.empty() &&
           type_impls_interfaces.empty() &&
           type_impls_named_constraints.empty() &&
           rewrite_constraints.empty() && !other_requirements;
  }

  friend auto operator==(const FacetTypeInfo& lhs, const FacetTypeInfo& rhs)
      -> bool {
    return lhs.extend_constraints == rhs.extend_constraints &&
           lhs.self_impls_constraints == rhs.self_impls_constraints &&
           lhs.extend_named_constraints == rhs.extend_named_constraints &&
           lhs.self_impls_named_constraints ==
               rhs.self_impls_named_constraints &&
           lhs.type_impls_interfaces == rhs.type_impls_interfaces &&
           lhs.type_impls_named_constraints ==
               rhs.type_impls_named_constraints &&
           lhs.rewrite_constraints == rhs.rewrite_constraints &&
           lhs.other_requirements == rhs.other_requirements;
  }
};

constexpr FacetTypeInfo::RewriteConstraint
    FacetTypeInfo::RewriteConstraint::None = {.lhs_id = InstId::None,
                                              .rhs_id = InstId::None};

using FacetTypeInfoStore =
    CanonicalValueStore<FacetTypeId, FacetTypeInfo, Tag<CheckIRId>>;

struct IdentifiedFacetTypeKey {
  FacetTypeId facet_type_id;
  ConstantId self_const_id;
  // Inside a named constraint, each identification of the `Self` facet type can
  // be unique, as it can be modified by each require declaration seen so far.
  // Uses -1 for identifying a facet type with a self-type from outside the
  // definition of an named constraint.
  int32_t num_require_impls = -1;

  friend auto operator==(const IdentifiedFacetTypeKey& lhs,
                         const IdentifiedFacetTypeKey& rhs) -> bool = default;
};

// The IdentifiedFacetType represents all of the interfaces required by a facet
// type against a given Self type, and any other types it constrains. The order
// of the interfaces is fixed for a given facet type, and can thus be used as a
// key for storing and finding witnesses or other data associated with a facet
// type.
struct IdentifiedFacetType {
  // A requirement that `self_facet_value` implements the `specific_interface`.
  struct RequiredImpl {
    ConstantId self_facet_value;
    SpecificInterface specific_interface;

    friend auto operator==(const RequiredImpl& lhs, const RequiredImpl& rhs)
        -> bool = default;
  };

  IdentifiedFacetType(IdentifiedFacetTypeKey key, bool partially_identified,
                      llvm::ArrayRef<RequiredImpl> extends,
                      llvm::ArrayRef<RequiredImpl> self_impls);

  // The order here defines the order of impl witnesses for this facet type.
  auto required_impls() const -> llvm::ArrayRef<RequiredImpl> {
    return required_impls_;
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

  auto partially_identified() const -> bool {
    return key_.num_require_impls >= 0;
  }

  auto GetAsKey() const -> IdentifiedFacetTypeKey { return key_; }

 private:
  IdentifiedFacetTypeKey key_;

  // Requirements that a facet value implements an interface, mentioned
  // explicitly in the facet type expression or transitively through a named
  // constraint. Sorted and deduplicated.
  llvm::SmallVector<RequiredImpl> required_impls_;

  // The single interface from `required_impls` to implement if this is
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

using IdentifiedFacetTypeStore =
    CanonicalValueStore<IdentifiedFacetTypeId, IdentifiedFacetTypeKey,
                        Tag<CheckIRId>, IdentifiedFacetType>;

// See common/hashing.h.
inline auto CarbonHashValue(const FacetTypeInfo& value, uint64_t seed)
    -> HashCode {
  Hasher hasher(seed);
  hasher.HashArray(llvm::ArrayRef(value.extend_constraints));
  hasher.HashArray(llvm::ArrayRef(value.self_impls_constraints));
  hasher.HashArray(llvm::ArrayRef(value.extend_named_constraints));
  hasher.HashArray(llvm::ArrayRef(value.self_impls_named_constraints));
  hasher.HashArray(llvm::ArrayRef(value.rewrite_constraints));
  hasher.HashRaw(value.other_requirements);
  return static_cast<HashCode>(hasher);
}

// Given an array of witnesses, sorts them to match the ordering of the specific
// interfaces in the IdentifiedFacetType that produced the witness set, which is
// the canonical witness order, and returns the resulting block ID. This assumes
// witnesses have already been deduplicated, and do not contain errors, because
// it's mainly for imports.
auto AddCanonicalWitnessesBlock(File& sem_ir,
                                llvm::SmallVector<InstId>& witnesses)
    -> InstBlockId;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_FACET_TYPE_INFO_H_
