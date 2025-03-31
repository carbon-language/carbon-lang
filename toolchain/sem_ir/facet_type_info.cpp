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

auto FacetTypeInfo::Canonicalize() -> void {
  SortAndDeduplicate(impls_constraints, ImplsLess);
  SortAndDeduplicate(rewrite_constraints, RewriteLess);
}

auto FacetTypeInfo::Print(llvm::raw_ostream& out) const -> void {
  out << "{";
  llvm::ListSeparator outer_sep("; ");

  if (!impls_constraints.empty()) {
    out << outer_sep << "impls interface: ";
    llvm::ListSeparator sep;
    for (ImplsConstraint req : impls_constraints) {
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

auto IdentifiedFacetType::CanonicalizeRequiredInterfaces() -> void {
  SortAndDeduplicate(required_interfaces_, RequiredLess);
}

}  // namespace Carbon::SemIR
