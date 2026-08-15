// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/identified_facet_type.h"

#include "toolchain/base/canonical_value_store_impl.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/base/value_store_impl.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

template <typename T>
using LessThanFn = llvm::function_ref<auto(const T&, const T&)->bool>;

template <typename VecT>
static auto SortAndDeduplicate(VecT& vec,
                               LessThanFn<typename VecT::value_type> compare)
    -> void {
  llvm::sort(vec, compare);
  vec.erase(llvm::unique(vec), vec.end());
}

// Canonically ordered by the numerical ids. Matches the order of RequiredLess
// when the `self_facet_value` is the same for all interfaces.
static auto InterfaceLess(const SpecificInterface& lhs,
                          const SpecificInterface& rhs) -> bool {
  return std::tie(lhs.interface_id.index, lhs.specific_id.index) <
         std::tie(rhs.interface_id.index, rhs.specific_id.index);
}

// Canonically ordered by the numerical ids.
static auto RequiredLess(const IdentifiedFacetType::RequiredImpl& lhs,
                         const IdentifiedFacetType::RequiredImpl& rhs) -> bool {
  return std::tie(lhs.self_facet_value.index,
                  lhs.specific_interface.interface_id.index,
                  lhs.specific_interface.specific_id.index) <
         std::tie(rhs.self_facet_value.index,
                  rhs.specific_interface.interface_id.index,
                  rhs.specific_interface.specific_id.index);
}

IdentifiedFacetType::IdentifiedFacetType(
    IdentifiedFacetTypeKey key, bool partially_identified,
    llvm::ArrayRef<RequiredImpl> extends,
    llvm::ArrayRef<RequiredImpl> self_impls)
    : key_(key) {
  required_impls_.reserve(extends.size() + self_impls.size());
  llvm::append_range(required_impls_, extends);
  SortAndDeduplicate(required_impls_, RequiredLess);

  // If there's a single extended interface then we present as that interface.
  // Otherwise, we record the number extended interfaces.
  if (required_impls_.size() == 1) {
    interface_id_ = required_impls_.front().specific_interface.interface_id;
    specific_id_ = required_impls_.front().specific_interface.specific_id;
  } else {
    interface_id_ = InterfaceId::None;
    num_interface_to_impl_ = required_impls_.size();
  }

  llvm::append_range(required_impls_, self_impls);
  SortAndDeduplicate(required_impls_, RequiredLess);

  if (partially_identified) {
    // This marks the IdentifiedFacetType as being partially identified, and
    // prevents the key from colliding with a fully identified facet type, or
    // with other partially (but differently) identified facet types, with the
    // same constituents but a more complete set of required interfaces.
    key_.num_require_impls = required_impls_.size();
  }
}

auto AddCanonicalWitnessesBlock(File& sem_ir,
                                llvm::SmallVector<InstId>& witnesses)
    -> InstBlockId {
  // Small blocks don't need to be sorted.
  if (witnesses.size() <= 1) {
    return sem_ir.inst_blocks().AddCanonical(witnesses);
  }

  llvm::SmallVector<std::pair<SpecificInterface, InstId>> sortable;
  sortable.reserve(witnesses.size());

  // Produce the sorted order based on the witness's SpecificInterface.
  for (auto witness_id : witnesses) {
    auto inst = sem_ir.insts().Get(witness_id);
    CARBON_KIND_SWITCH(inst) {
      case CARBON_KIND(CustomWitness witness): {
        sortable.push_back({sem_ir.specific_interfaces().Get(
                                witness.query_specific_interface_id),
                            witness_id});
        break;
      }
      case CARBON_KIND(ImplWitness witness): {
        auto table =
            sem_ir.insts().GetAs<ImplWitnessTable>(witness.witness_table_id);
        sortable.push_back(
            {sem_ir.impls().Get(table.impl_id).interface, witness_id});
        break;
      }
      case CARBON_KIND(LookupImplWitness witness): {
        sortable.push_back({sem_ir.specific_interfaces().Get(
                                witness.query_specific_interface_id),
                            witness_id});
        break;
      }
      default:
        CARBON_FATAL("Unhandled inst: {0}", inst);
    }
  }
  // This matches the sort order of IdentifiedFacetType::required_interfaces,
  // which is the order of the witnesses returned from impl lookup, and is
  // canonical order in which the witnesses must appear for a given facet type
  // so that ImplWitnessAccess can find the appropriate witness.
  llvm::sort(sortable, [](auto& lhs, auto& rhs) {
    return InterfaceLess(lhs.first, rhs.first);
  });

  // Update the original list with the new order (reusing to avoid an
  // allocation).
  for (auto [witness_id, sortable_entry] :
       llvm::zip_equal(witnesses, sortable)) {
    witness_id = sortable_entry.second;
  }

  return sem_ir.inst_blocks().AddCanonical(witnesses);
}

}  // namespace Carbon::SemIR
