// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_CPP_OVERLOAD_SET_H_
#define CARBON_TOOLCHAIN_SEM_IR_CPP_OVERLOAD_SET_H_

#include "clang/AST/UnresolvedSet.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "common/ostream.h"
#include "toolchain/base/value_store.h"
#include "toolchain/sem_ir/ids.h"

namespace clang {
class CXXRecordDecl;
}  // namespace clang

namespace Carbon::SemIR {

// An overloaded C++ function.
struct CppOverloadSet : public Printable<CppOverloadSet> {
  // Information about operator rewrites to consider when adding operator
  // functions to a candidate set.
  //
  // This mirrors `clang::OverloadCandidateSet::OperatorRewriteInfo` so that
  // this header doesn't need `clang/Sema/Overload.h`; the use sites construct
  // the Clang type from these fields.
  struct OperatorRewriteInfo {
    // The original operator as written in the source.
    clang::OverloadedOperatorKind original_operator = clang::OO_None;
    // The source location of the operator.
    clang::SourceLocation op_loc;
    // Whether we should include rewritten candidates in the overload set.
    bool allow_rewritten_candidates = false;
  };

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
  OperatorRewriteInfo operator_rewrite_info;

  auto Print(llvm::raw_ostream& out) const -> void {
    out << "name: " << name_id << ", parent_scope: " << parent_scope_id;
  }
};

using CppOverloadSetStore =
    ValueStore<CppOverloadSetId, CppOverloadSet, Tag<CheckIRId>>;

}  // namespace Carbon::SemIR

namespace Carbon {
extern template class ValueStore<SemIR::CppOverloadSetId, SemIR::CppOverloadSet,
                                 Tag<SemIR::CheckIRId>>;
}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEM_IR_CPP_OVERLOAD_SET_H_
