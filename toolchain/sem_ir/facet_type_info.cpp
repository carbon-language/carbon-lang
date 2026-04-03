// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/facet_type_info.h"

#include <tuple>

#include "toolchain/base/kind_switch.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

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

// Canonically ordered by the numerical ids.
static auto InterfaceLess(const SpecificInterface& lhs,
                          const SpecificInterface& rhs) -> bool {
  return std::tie(lhs.interface_id.index, lhs.specific_id.index) <
         std::tie(rhs.interface_id.index, rhs.specific_id.index);
}

// Canonically ordered by the numerical ids.
static auto RewriteLess(const FacetTypeInfo::RewriteConstraint& lhs,
                        const FacetTypeInfo::RewriteConstraint& rhs) -> bool {
  return std::tie(lhs.lhs_id.index, lhs.rhs_id.index) <
         std::tie(rhs.lhs_id.index, rhs.rhs_id.index);
}

// Canonically ordered by the numerical ids.
static auto NamedConstraintLess(const SpecificNamedConstraint& lhs,
                                const SpecificNamedConstraint& rhs) -> bool {
  return std::tie(lhs.named_constraint_id.index, lhs.specific_id.index) <
         std::tie(rhs.named_constraint_id.index, rhs.specific_id.index);
}

// Canonically ordered by the numerical ids.
static auto TypeImplsInterfaceLess(const FacetTypeInfo::TypeImplsInterface& lhs,
                                   const FacetTypeInfo::TypeImplsInterface& rhs)
    -> bool {
  return std::tie(lhs.self_type.index,
                  lhs.specific_interface.interface_id.index,
                  lhs.specific_interface.specific_id.index) <
         std::tie(rhs.self_type.index,
                  rhs.specific_interface.interface_id.index,
                  rhs.specific_interface.specific_id.index);
}

// Canonically ordered by the numerical ids.
static auto TypeImplsNamedConstraintLess(
    const FacetTypeInfo::TypeImplsNamedConstraint& lhs,
    const FacetTypeInfo::TypeImplsNamedConstraint& rhs) -> bool {
  return std::tie(lhs.self_type.index,
                  lhs.specific_named_constraint.named_constraint_id.index,
                  lhs.specific_named_constraint.specific_id.index) <
         std::tie(rhs.self_type.index,
                  rhs.specific_named_constraint.named_constraint_id.index,
                  rhs.specific_named_constraint.specific_id.index);
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

// Assuming both `a` and `b` are sorted and deduplicated, replaces `a` with `a -
// b` as sets. Assumes there are few elements between them.
template <typename VecT>
static auto SubtractSorted(VecT& a, const VecT& b,
                           LessThanFn<typename VecT::value_type> compare)
    -> void {
  using Iter = VecT::iterator;
  Iter a_iter = a.begin();
  Iter a_end = a.end();
  using ConstIter = VecT::const_iterator;
  ConstIter b_iter = b.begin();
  ConstIter b_end = b.end();
  // Advance the iterator pointing to the smaller element until we find a match.
  while (a_iter != a_end && b_iter != b_end) {
    if (compare(*a_iter, *b_iter)) {
      ++a_iter;
    } else if (compare(*b_iter, *a_iter)) {
      ++b_iter;
    } else {
      break;
    }
  }
  if (a_iter == a_end || b_iter == b_end) {
    // Nothing to remove from `a`.
    return;
  }
  // Found a match, switch to removing elements of `a`.
  CARBON_DCHECK(*a_iter == *b_iter);
  // We copy the elements we want to keep to `*a_new_end`, and skip the elements
  // of `a` that match something in `b`.
  Iter a_new_end = a_iter;
  ++a_iter;
  ++b_iter;
  while (a_iter != a_end && b_iter != b_end) {
    if (compare(*a_iter, *b_iter)) {
      *a_new_end = *a_iter;
      ++a_new_end;
      ++a_iter;
    } else if (compare(*b_iter, *a_iter)) {
      ++b_iter;
    } else {
      CARBON_DCHECK(*a_iter == *b_iter);
      ++a_iter;
      ++b_iter;
    }
  }
  // Keep the remaining elements of `a`, if any.
  for (; a_iter != a_end; ++a_iter) {
    *a_new_end = *a_iter;
    ++a_new_end;
  }
  // Shrink `a` by the number of elements that we skipped since they matched
  // something in `b`.
  a.erase(a_new_end, a_end);
}

template <typename VecT>
static auto CombineVectors(VecT& vec, const VecT& lhs, const VecT& rhs) {
  vec.reserve(lhs.size() + rhs.size());
  llvm::append_range(vec,
                     llvm::concat<const typename VecT::value_type>(lhs, rhs));
}

auto FacetTypeInfo::Combine(const FacetTypeInfo& lhs, const FacetTypeInfo& rhs)
    -> FacetTypeInfo {
  FacetTypeInfo info;
  CombineVectors(info.extend_constraints, lhs.extend_constraints,
                 rhs.extend_constraints);
  CombineVectors(info.self_impls_constraints, lhs.self_impls_constraints,
                 rhs.self_impls_constraints);
  CombineVectors(info.extend_named_constraints, lhs.extend_named_constraints,
                 rhs.extend_named_constraints);
  CombineVectors(info.self_impls_named_constraints,
                 lhs.self_impls_named_constraints,
                 rhs.self_impls_named_constraints);
  CombineVectors(info.type_impls_interfaces, lhs.type_impls_interfaces,
                 rhs.type_impls_interfaces);
  CombineVectors(info.type_impls_named_constraints,
                 lhs.type_impls_named_constraints,
                 rhs.type_impls_named_constraints);
  CombineVectors(info.rewrite_constraints, lhs.rewrite_constraints,
                 rhs.rewrite_constraints);
  info.other_requirements = lhs.other_requirements || rhs.other_requirements;
  return info;
}

auto FacetTypeInfo::Canonicalize() -> void {
  SortAndDeduplicate(extend_constraints, InterfaceLess);
  SortAndDeduplicate(self_impls_constraints, InterfaceLess);
  SubtractSorted(self_impls_constraints, extend_constraints, InterfaceLess);
  SortAndDeduplicate(extend_named_constraints, NamedConstraintLess);
  SortAndDeduplicate(self_impls_named_constraints, NamedConstraintLess);
  SubtractSorted(self_impls_named_constraints, extend_named_constraints,
                 NamedConstraintLess);
  SortAndDeduplicate(type_impls_interfaces, TypeImplsInterfaceLess);
  SortAndDeduplicate(type_impls_named_constraints,
                     TypeImplsNamedConstraintLess);
  SortAndDeduplicate(rewrite_constraints, RewriteLess);
}

auto FacetTypeInfo::Print(llvm::raw_ostream& out) const -> void {
  out << "{";
  llvm::ListSeparator outer_sep("; ");

  if (!extend_constraints.empty()) {
    out << outer_sep << "extends interface: ";
    llvm::ListSeparator sep;
    for (auto req : extend_constraints) {
      out << sep << req.interface_id;
      if (req.specific_id.has_value()) {
        out << "(" << req.specific_id << ")";
      }
    }
  }

  if (!self_impls_constraints.empty()) {
    out << outer_sep << "self impls interface: ";
    llvm::ListSeparator sep;
    for (auto req : self_impls_constraints) {
      out << sep << req.interface_id;
      if (req.specific_id.has_value()) {
        out << "(" << req.specific_id << ")";
      }
    }
  }

  if (!extend_named_constraints.empty()) {
    out << outer_sep << "extends named constraint: ";
    llvm::ListSeparator sep;
    for (auto extend : extend_named_constraints) {
      out << sep << extend.named_constraint_id;
      if (extend.specific_id.has_value()) {
        out << "(" << extend.specific_id << ")";
      }
    }
  }

  if (!self_impls_named_constraints.empty()) {
    out << outer_sep << "self impls named constraint: ";
    llvm::ListSeparator sep;
    for (auto self_impls : self_impls_named_constraints) {
      out << sep << self_impls.named_constraint_id;
      if (self_impls.specific_id.has_value()) {
        out << "(" << self_impls.specific_id << ")";
      }
    }
  }

  if (!type_impls_interfaces.empty()) {
    out << outer_sep << "type impls interface: ";
    llvm::ListSeparator sep;
    for (const auto& type_impls : type_impls_interfaces) {
      out << sep << type_impls.self_type;
      out << " impls " << type_impls.specific_interface.interface_id;
      if (type_impls.specific_interface.specific_id.has_value()) {
        out << "(" << type_impls.specific_interface.specific_id << ")";
      }
    }
  }

  if (!type_impls_named_constraints.empty()) {
    out << outer_sep << "type impls interface: ";
    llvm::ListSeparator sep;
    for (const auto& type_impls : type_impls_named_constraints) {
      out << sep << type_impls.self_type;
      out << " impls "
          << type_impls.specific_named_constraint.named_constraint_id;
      if (type_impls.specific_named_constraint.specific_id.has_value()) {
        out << "(" << type_impls.specific_named_constraint.specific_id << ")";
      }
    }
  }

  if (!rewrite_constraints.empty()) {
    out << outer_sep << "rewrites: ";
    llvm::ListSeparator sep;
    for (auto req : rewrite_constraints) {
      out << sep << req.lhs_id << "=" << req.rhs_id;
    }
  }

  if (other_requirements) {
    out << outer_sep << "+ TODO requirements";
  }

  out << "}";
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
