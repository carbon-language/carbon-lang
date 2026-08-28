// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_DECLARED_FACET_TYPE_H_
#define CARBON_TOOLCHAIN_SEM_IR_DECLARED_FACET_TYPE_H_

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
struct DeclaredFacetType : Printable<DeclaredFacetType> {
  // Returns a DeclaredFacetType that combines `lhs` and `rhs`. It is not
  // canonicalized, so that it can be further modified by the caller if desired.
  static auto Combine(const DeclaredFacetType& lhs,
                      const DeclaredFacetType& rhs) -> DeclaredFacetType;

  // Returns a DeclaredFacetType that only contains constraints that are
  // extended by the facet type. It is not canonicalized, so that it can be
  // further modified by the caller if desired.
  static auto ExtendedOnly(const DeclaredFacetType& declared_facet_type)
      -> DeclaredFacetType;

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
  auto TryAsSingleExtend() const -> std::optional<SingleExtendFacetType>;

  // Returns whether the facet type has no constraints, making it the facet type
  // version of `TypeType`.
  auto HasNoConstraints() const -> bool;

  // Returns whether the facet type only contains constraints that are extended
  // by the facet type. If true, `ExtendedOnly()` would be a no-op.
  auto IsExtendedOnly() const -> bool;

  friend auto operator==(const DeclaredFacetType& lhs,
                         const DeclaredFacetType& rhs) -> bool = default;
};

constexpr DeclaredFacetType::RewriteConstraint
    DeclaredFacetType::RewriteConstraint::None = {.lhs_id = InstId::None,
                                                  .rhs_id = InstId::None};

using DeclaredFacetTypeStore =
    CanonicalValueStore<DeclaredFacetTypeId, DeclaredFacetType, Tag<CheckIRId>>;

// See common/hashing.h.
inline auto CarbonHashValue(const DeclaredFacetType& value, uint64_t seed)
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

}  // namespace Carbon::SemIR

namespace Carbon {
extern template class CanonicalValueStore<SemIR::DeclaredFacetTypeId,
                                          SemIR::DeclaredFacetType,
                                          Tag<SemIR::CheckIRId>>;
extern template class ValueStore<SemIR::DeclaredFacetTypeId,
                                 SemIR::DeclaredFacetType,
                                 Tag<SemIR::CheckIRId>>;
}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEM_IR_DECLARED_FACET_TYPE_H_
