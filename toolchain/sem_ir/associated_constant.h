// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_ASSOCIATED_CONSTANT_H_
#define CARBON_TOOLCHAIN_SEM_IR_ASSOCIATED_CONSTANT_H_

#include "common/ostream.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// An associated constant entity. For example:
//
//   interface I {
//     let AssocConst:! type;
//    }
//
// TODO: This overlaps a lot with EntityName and EntityWithParamsBase.
// Investigate ways of factoring out the common parts.
struct AssociatedConstant : public Printable<AssociatedConstant> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{" << "name: " << name_id << ", parent_scope: " << parent_scope_id
        << "}";
  }

  // The entity's name.
  NameId name_id;

  // The parent scope.
  NameScopeId parent_scope_id;

  // If this is a generic entity, information about the generic.
  GenericId generic_id;

  // The declaration of this associated constant.
  InstId decl_id;

  // The default value of the constant.
  InstId default_value_id = InstId::Invalid;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_ASSOCIATED_CONSTANT_H_
