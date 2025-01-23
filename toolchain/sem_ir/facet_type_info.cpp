// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/facet_type_info.h"

namespace Carbon::SemIR {

template <typename VecT>
static auto SortAndDeduplicate(VecT& vec) -> void {
  llvm::sort(vec);
  vec.erase(llvm::unique(vec), vec.end());
}

auto FacetTypeInfo::Canonicalize() -> void {
  SortAndDeduplicate(impls_constraints);
  SortAndDeduplicate(rewrite_constraints);
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

}  // namespace Carbon::SemIR
