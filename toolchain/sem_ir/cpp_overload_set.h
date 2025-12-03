// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_CPP_OVERLOAD_SET_H_
#define CARBON_TOOLCHAIN_SEM_IR_CPP_OVERLOAD_SET_H_

#include "clang/AST/Decl.h"
#include "clang/AST/UnresolvedSet.h"
#include "clang/Sema/Overload.h"
#include "common/ostream.h"
#include "toolchain/base/value_store.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// An overloaded C++ function.
struct CppOverloadSet : public Printable<CppOverloadSet> {
  // The function's name.
  NameId name_id;

  // The parent scope.
  NameScopeId parent_scope_id;

  // The naming class in the name lookup that found this overload set.
  clang::CXXRecordDecl* naming_class;

  // List of all named decls found at name lookup.
  // TODO: Find a good small size for the UnresolvedSet<size> or rework how we
  // store the candidates.
  clang::UnresolvedSet<4> candidate_functions;

  /// Information about operator rewrites to consider when adding operator
  /// functions to a candidate set.
  clang::OverloadCandidateSet::OperatorRewriteInfo operator_rewrite_info;

  auto Print(llvm::raw_ostream& out) const -> void {
    out << "name: " << name_id << ", parent_scope: " << parent_scope_id;
  }
};

using CppOverloadSetStore = ValueStore<CppOverloadSetId, CppOverloadSet>;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_CPP_OVERLOAD_SET_H_
