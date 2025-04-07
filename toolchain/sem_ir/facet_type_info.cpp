// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/facet_type_info.h"

namespace Carbon::SemIR {

template <typename VecT, typename CompareT>
static auto SortAndDeduplicate(VecT& vec, CompareT compare) -> void {
  llvm::sort(vec, compare);
  vec.erase(llvm::unique(vec), vec.end());
}

// Canonically ordered by the numerical ids.
static auto ImplsLess(const FacetTypeInfo::ImplsConstraint& lhs,
                      const FacetTypeInfo::ImplsConstraint& rhs) -> bool {
  return std::tie(lhs.interface_id.index, lhs.specific_id.index) <
         std::tie(rhs.interface_id.index, rhs.specific_id.index);
}

// Canonically ordered by the numerical ids.
static auto RewriteLess(const FacetTypeInfo::RewriteConstraint& lhs,
                        const FacetTypeInfo::RewriteConstraint& rhs) -> bool {
  return std::tie(lhs.lhs_const_id.index, lhs.rhs_const_id.index) <
         std::tie(rhs.lhs_const_id.index, rhs.rhs_const_id.index);
}

// Canonically ordered by the numerical ids.
static auto RequiredLess(const IdentifiedFacetType::RequiredInterface& lhs,
                         const IdentifiedFacetType::RequiredInterface& rhs)
    -> bool {
  return std::tie(lhs.interface_id.index, lhs.specific_id.index) <
         std::tie(rhs.interface_id.index, rhs.specific_id.index);
}

// Assuming both `a` and `b` are sorted and deduplicated, replaces `a` with `a -
// b` as sets. Assumes there are few elements between them.
static auto SubtractSorted(
    llvm::SmallVector<FacetTypeInfo::ImplsConstraint>& a,
    const llvm::SmallVector<FacetTypeInfo::ImplsConstraint>& b) -> void {
  using Iter = llvm::SmallVector<FacetTypeInfo::ImplsConstraint>::iterator;
  Iter a_iter = a.begin();
  Iter a_end = a.end();
  using ConstIter =
      llvm::SmallVector<FacetTypeInfo::ImplsConstraint>::const_iterator;
  ConstIter b_iter = b.begin();
  ConstIter b_end = b.end();
  // Advance the iterator pointing to the smaller element until we find a match.
  while (a_iter != a_end && b_iter != b_end) {
    if (ImplsLess(*a_iter, *b_iter)) {
      ++a_iter;
    } else if (ImplsLess(*b_iter, *a_iter)) {
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
    if (ImplsLess(*a_iter, *b_iter)) {
      *a_new_end = *a_iter;
      ++a_new_end;
      ++a_iter;
    } else if (ImplsLess(*b_iter, *a_iter)) {
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

auto FacetTypeInfo::Combine(const FacetTypeInfo& lhs, const FacetTypeInfo& rhs)
    -> FacetTypeInfo {
  FacetTypeInfo info = {.other_requirements = false};
  info.extend_constraints.reserve(lhs.extend_constraints.size() +
                                  rhs.extend_constraints.size());
  llvm::append_range(info.extend_constraints, lhs.extend_constraints);
  llvm::append_range(info.extend_constraints, rhs.extend_constraints);
  info.self_impls_constraints.reserve(lhs.self_impls_constraints.size() +
                                      rhs.self_impls_constraints.size());
  llvm::append_range(info.self_impls_constraints, lhs.self_impls_constraints);
  llvm::append_range(info.self_impls_constraints, rhs.self_impls_constraints);
  info.rewrite_constraints.reserve(lhs.rewrite_constraints.size() +
                                   rhs.rewrite_constraints.size());
  llvm::append_range(info.rewrite_constraints, lhs.rewrite_constraints);
  llvm::append_range(info.rewrite_constraints, rhs.rewrite_constraints);
  info.other_requirements |= lhs.other_requirements;
  info.other_requirements |= rhs.other_requirements;
  return info;
}

auto FacetTypeInfo::Canonicalize() -> void {
  SortAndDeduplicate(extend_constraints, ImplsLess);
  SortAndDeduplicate(self_impls_constraints, ImplsLess);
  SubtractSorted(self_impls_constraints, extend_constraints);
  SortAndDeduplicate(rewrite_constraints, RewriteLess);
}

auto FacetTypeInfo::Print(llvm::raw_ostream& out) const -> void {
  out << "{";
  llvm::ListSeparator outer_sep("; ");

  if (!extend_constraints.empty()) {
    out << outer_sep << "extends interface: ";
    llvm::ListSeparator sep;
    for (ImplsConstraint req : extend_constraints) {
      out << sep << req.interface_id;
      if (req.specific_id.has_value()) {
        out << "(" << req.specific_id << ")";
      }
    }
  }

  if (!self_impls_constraints.empty()) {
    out << outer_sep << "self impls interface: ";
    llvm::ListSeparator sep;
    for (ImplsConstraint req : self_impls_constraints) {
      out << sep << req.interface_id;
      if (req.specific_id.has_value()) {
        out << "(" << req.specific_id << ")";
      }
    }
  }

  if (!rewrite_constraints.empty()) {
    out << outer_sep << "rewrites: ";
    llvm::ListSeparator sep;
    for (RewriteConstraint req : rewrite_constraints) {
      out << sep << req.lhs_const_id << "=" << req.rhs_const_id;
    }
  }

  if (other_requirements) {
    out << outer_sep << "+ TODO requirements";
  }

  out << "}";
}

IdentifiedFacetType::IdentifiedFacetType(
    llvm::ArrayRef<RequiredInterface> extends,
    llvm::ArrayRef<RequiredInterface> self_impls) {
  if (extends.size() == 1) {
    interface_id_ = extends.front().interface_id;
    specific_id_ = extends.front().specific_id;
  } else {
    interface_id_ = InterfaceId::None;
    num_interface_to_impl_ = extends.size();
  }

  required_interfaces_.reserve(extends.size() + self_impls.size());
  llvm::append_range(required_interfaces_, extends);
  llvm::append_range(required_interfaces_, self_impls);
  SortAndDeduplicate(required_interfaces_, RequiredLess);
}

}  // namespace Carbon::SemIR
