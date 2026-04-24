// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_NAMED_CONSTRAINT_H_
#define CARBON_TOOLCHAIN_SEM_IR_NAMED_CONSTRAINT_H_

#include "toolchain/base/value_store.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// Named constraint-specific fields.
struct NamedConstraintFields {
  // The following members are set at the `{` of the constraint definition.

  // The constraint scopes. The scope-without-self contains the symbolic `Self`
  // entity, which is then a generic binding of the generic-with-self. The
  // scope-with-self contains the rest of the entities in the constraint.
  NameScopeId scope_without_self_id = NameScopeId::None;
  NameScopeId scope_with_self_id = NameScopeId::None;
  // The block of instructions outside the constraint-with-self. This is where
  // the `Self` instruction can be constructed.
  InstBlockId body_block_without_self_id = InstBlockId::None;
  // The constraint-with-self generic, where the `Self` is a parameter to the
  // generic. This generic contains all the associated entities of the
  // interface.
  GenericId generic_with_self_id = GenericId::None;
  // The first block of the constraint-with-self body.
  InstBlockId body_block_with_self_id = InstBlockId::None;
  // The implicit `Self` parameter. This is a SymbolicBinding instruction.
  InstId self_param_id = InstId::None;

  // The following members are set at the `}` of the constraint definition.

  RequireImplsBlockId require_impls_block_id = RequireImplsBlockId::None;
  bool complete = false;
};

// A named constraint. See EntityWithParamsBase regarding the inheritance here.
struct NamedConstraint : public EntityWithParamsBase,
                         public NamedConstraintFields,
                         public Printable<NamedConstraint> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{";
    PrintBaseFields(out);
    out << ", require_impls_block_id: " << require_impls_block_id;
    out << "}";
  }

  // This is false until we reach the `}` of the constraint definition.
  auto is_complete() const -> bool { return complete; }

  // Determines whether we're currently defining the constraint. This is true
  // between the braces of the constraint.
  auto is_being_defined() const -> bool {
    return has_definition_started() && !is_complete();
  }
};

using NamedConstraintStore =
    ValueStore<NamedConstraintId, NamedConstraint, Tag<CheckIRId>>;

}  // namespace Carbon::SemIR

namespace Carbon {
extern template class ValueStore<SemIR::NamedConstraintId,
                                 SemIR::NamedConstraint, Tag<SemIR::CheckIRId>>;
}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEM_IR_NAMED_CONSTRAINT_H_
