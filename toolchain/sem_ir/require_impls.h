// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_REQUIRE_IMPLS_H_
#define CARBON_TOOLCHAIN_SEM_IR_REQUIRE_IMPLS_H_

#include "toolchain/base/block_value_store.h"
#include "toolchain/base/canonical_value_store.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// An interface requirement from an interface or named constraint, written
// `require T impls Z`.
//
// While this comes from a `require` declaration, it is not an Entity like most
// other declarations, with a name and parameters, so it does not inherit
// EntityWithParamsBase.
struct RequireImpls : Printable<RequireImpls> {
  // The self-type which must implement a given facet type.
  TypeInstId self_id;
  // Evaluates to the `FacetType` that the self-type must implement.
  TypeInstId facet_type_inst_id;
  // The `FacetTypeInfo` derived from the `facet_type_inst_id` instruction.
  FacetTypeId facet_type_id;

  // The location of the `require` declaration.
  InstId decl_id;
  // The interface or named constraint which contains the `require` declaration.
  NameScopeId parent_scope_id;
  // The instructions that make up the `require` declaration. It will contain
  // instructions that provide the `self_id` and `facet_type_inst_id`.
  InstBlockId body_block_id;
  // A `require` declaration is always generic over `Self` since it's inside an
  // interface or named constraint definition.
  GenericId generic_id;

  auto Print(llvm::raw_ostream& out) const -> void {
    out << '{';
    out << "self_id: " << self_id
        << ", facet_type_inst_id: " << facet_type_inst_id
        << ", parent_scope: " << parent_scope_id;
    out << '}';
  }
};

using RequireImplsStore = ValueStore<RequireImplsId, RequireImpls>;

using RequireImplsBlockStore =
    BlockValueStore<RequireImplsBlockId, RequireImplsId>;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_REQUIRE_IMPLS_H_
