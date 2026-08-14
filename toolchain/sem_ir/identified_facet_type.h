// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_IDENTIFIED_FACET_TYPE_H_
#define CARBON_TOOLCHAIN_SEM_IR_IDENTIFIED_FACET_TYPE_H_

#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/canonical_value_store.h"
#include "toolchain/sem_ir/declared_facet_type.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/specific_interface.h"

namespace Carbon::SemIR {

class File;

struct IdentifiedFacetTypeKey {
  DeclaredFacetTypeId declared_facet_type_id;
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

// Given an array of witnesses, sorts them to match the ordering of the specific
// interfaces in the IdentifiedFacetType that produced the witness set, which is
// the canonical witness order, and returns the resulting block ID. This assumes
// witnesses have already been deduplicated, and do not contain errors, because
// it's mainly for imports.
auto AddCanonicalWitnessesBlock(File& sem_ir,
                                llvm::SmallVector<InstId>& witnesses)
    -> InstBlockId;

}  // namespace Carbon::SemIR

namespace Carbon {
extern template class CanonicalValueStore<
    SemIR::IdentifiedFacetTypeId, SemIR::IdentifiedFacetTypeKey,
    Tag<SemIR::CheckIRId>, SemIR ::IdentifiedFacetType>;
extern template class ValueStore<SemIR::IdentifiedFacetTypeId,
                                 SemIR::IdentifiedFacetType,
                                 Tag<SemIR::CheckIRId>>;
}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEM_IR_IDENTIFIED_FACET_TYPE_H_
