// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/facet_type_info.h"

#include <tuple>

#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

CARBON_DEFINE_ENUM_MASK_NAMES(BuiltinConstraintMask) {
  CARBON_BUILTIN_CONSTRAINT_MASK(CARBON_ENUM_MASK_NAME_STRING)
};

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
static auto ImplsLess(const FacetTypeInfo::ImplsConstraint& lhs,
                      const FacetTypeInfo::ImplsConstraint& rhs) -> bool {
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
static auto NamedConstraintsLess(const SpecificNamedConstraint& lhs,
                                 const SpecificNamedConstraint& rhs) -> bool {
  return std::tie(lhs.named_constraint_id.index, lhs.specific_id.index) <
         std::tie(rhs.named_constraint_id.index, rhs.specific_id.index);
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
  CombineVectors(info.rewrite_constraints, lhs.rewrite_constraints,
                 rhs.rewrite_constraints);
  info.builtin_constraint_mask =
      lhs.builtin_constraint_mask | rhs.builtin_constraint_mask;
  info.other_requirements = lhs.other_requirements || rhs.other_requirements;
  return info;
}

auto FacetTypeInfo::Canonicalize() -> void {
  SortAndDeduplicate(extend_constraints, ImplsLess);
  SortAndDeduplicate(self_impls_constraints, ImplsLess);
  SubtractSorted(self_impls_constraints, extend_constraints, ImplsLess);
  SortAndDeduplicate(extend_named_constraints, NamedConstraintsLess);
  SortAndDeduplicate(self_impls_named_constraints, NamedConstraintsLess);
  SubtractSorted(self_impls_named_constraints, extend_named_constraints,
                 NamedConstraintsLess);
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
      out << sep << req.lhs_id << "=" << req.rhs_id;
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

  if (!self_impls_constraints.empty()) {
    out << outer_sep << "self impls named constraint: ";
    llvm::ListSeparator sep;
    for (auto self_impls : self_impls_named_constraints) {
      out << sep << self_impls.named_constraint_id;
      if (self_impls.specific_id.has_value()) {
        out << "(" << self_impls.specific_id << ")";
      }
    }
  }

  if (!builtin_constraint_mask.empty()) {
    out << outer_sep << "builtin_constraint_mask: " << builtin_constraint_mask;
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
