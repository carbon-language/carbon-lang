// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_INTERFACE_H_
#define CARBON_TOOLCHAIN_SEM_IR_INTERFACE_H_

#include "toolchain/base/value_store.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// Interface-specific fields.
//
// TODO: Factor out the shared fields between InterfaceFields and
// NamedConstraintFields.
struct InterfaceFields {
  // The following members are set at the `{` of the interface definition.

  // The interface scopes.
  NameScopeId scope_without_self_id = NameScopeId::None;
  NameScopeId scope_with_self_id = NameScopeId::None;
  // The block of instructions outside the interface-with-self. This is where
  // the `Self` instruction can be constructed.
  InstBlockId body_block_without_self_id = InstBlockId::None;
  // The interface-with-self generic, where the `Self` is a parameter to the
  // generic. This generic contains all the associated entities of the
  // interface.
  GenericId generic_with_self_id = GenericId::None;
  // The first block of the interface-with-self body.
  // TODO: Handle control flow in the interface body, such as if-expressions.
  InstBlockId body_block_with_self_id = InstBlockId::None;
  // The implicit `Self` parameter. This is a SymbolicBinding instruction.
  InstId self_param_id = InstId::None;

  // The following members are set at the `}` of the interface definition.

  RequireImplsBlockId require_impls_block_id = RequireImplsBlockId::None;
  InstBlockId associated_entities_id = InstBlockId::None;
};

// An interface. See EntityWithParamsBase regarding the inheritance here.
struct Interface : public EntityWithParamsBase,
                   public InterfaceFields,
                   public Printable<Interface> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{";
    PrintBaseFields(out);
    out << ", require_impls_block_id: " << require_impls_block_id;
    out << "}";
  }

  // This is false until we reach the `}` of the interface definition.
  auto is_complete() const -> bool {
    return associated_entities_id.has_value();
  }

  // Determines whether we're currently defining the interface. This is true
  // between the braces of the interface.
  auto is_being_defined() const -> bool {
    return has_definition_started() && !is_complete();
  }
};

using InterfaceStore = ValueStore<InterfaceId, Interface, Tag<CheckIRId>>;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_INTERFACE_H_
