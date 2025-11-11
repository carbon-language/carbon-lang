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

#define CARBON_BUILTIN_CONSTRAINT_MASK(X)                  \
  /* Verifies types can use the builtin `type.destroy`. */ \
  X(TypeCanDestroy)

CARBON_DEFINE_RAW_ENUM_MASK(BuiltinConstraintMask, uint32_t) {
  CARBON_BUILTIN_CONSTRAINT_MASK(CARBON_RAW_ENUM_MASK_ENUMERATOR)
};

// Constraints that are produced by builtin functions.
//
// These constraints are not treated as full interfaces, and behave somewhat
// similarly to `type where .Self impls <builtin>` as an API. Similarly, `impl C
// as <BuiltinConstraint>` will be invalid because `impl` requires at least one
// extended interface.
class BuiltinConstraintMask
    : public CARBON_ENUM_MASK_BASE(BuiltinConstraintMask) {
 public:
  CARBON_BUILTIN_CONSTRAINT_MASK(CARBON_ENUM_MASK_CONSTANT_DECL)

  using EnumMaskBase::AsInt;
};

#define CARBON_BUILTIN_CONSTRAINT_MASK_WITH_TYPE(X) \
  CARBON_ENUM_MASK_CONSTANT_DEFINITION(BuiltinConstraintMask, X)
CARBON_BUILTIN_CONSTRAINT_MASK(CARBON_BUILTIN_CONSTRAINT_MASK_WITH_TYPE)
#undef CARBON_BUILTIN_CONSTRAINT_MASK_WITH_TYPE

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

  // These name constraints add interfaces as lookup contexts, if they are
  // extended in the named constraint.
  llvm::SmallVector<SpecificNamedConstraint> extend_named_constraints;
  // These name constraints don't add interfaces as lookup contexts.
  llvm::SmallVector<SpecificNamedConstraint> self_impls_named_constraints;

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

  BuiltinConstraintMask builtin_constraint_mask = BuiltinConstraintMask::None;

  // TODO: Add same-type constraints.
  // TODO: Remove once all requirements are supported.
  bool other_requirements = false;

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
        extend_named_constraints.empty() &&
        self_impls_named_constraints.empty() && rewrite_constraints.empty() &&
        builtin_constraint_mask.empty() && !other_requirements) {
      return extend_constraints.front();
    }
    return std::nullopt;
  }

  friend auto operator==(const FacetTypeInfo& lhs, const FacetTypeInfo& rhs)
      -> bool {
    return lhs.extend_constraints == rhs.extend_constraints &&
           lhs.self_impls_constraints == rhs.self_impls_constraints &&
           lhs.extend_named_constraints == rhs.extend_named_constraints &&
           lhs.self_impls_named_constraints ==
               rhs.self_impls_named_constraints &&
           lhs.rewrite_constraints == rhs.rewrite_constraints &&
           lhs.builtin_constraint_mask == rhs.builtin_constraint_mask &&
           lhs.other_requirements == rhs.other_requirements;
  }
};

constexpr FacetTypeInfo::RewriteConstraint
    FacetTypeInfo::RewriteConstraint::None = {.lhs_id = InstId::None,
                                              .rhs_id = InstId::None};

using FacetTypeInfoStore = CanonicalValueStore<FacetTypeId, FacetTypeInfo>;

// TODO: This should probably include `BuiltinConstraintMask`, allowing APIs to
// include builtin constraints where `RequireIdentifiedFacetType` is used.
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
  hasher.HashArray(llvm::ArrayRef(value.extend_constraints));
  hasher.HashArray(llvm::ArrayRef(value.self_impls_constraints));
  hasher.HashArray(llvm::ArrayRef(value.extend_named_constraints));
  hasher.HashArray(llvm::ArrayRef(value.self_impls_named_constraints));
  hasher.HashArray(llvm::ArrayRef(value.rewrite_constraints));
  hasher.HashRaw(value.builtin_constraint_mask);
  hasher.HashRaw(value.other_requirements);
  return static_cast<HashCode>(hasher);
}

// Declared:
// (Carbon::HashCode)  (value_ = 12557349131240970624)

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_FACET_TYPE_INFO_H_
