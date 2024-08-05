// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_INTERFACE_H_
#define CARBON_TOOLCHAIN_SEM_IR_INTERFACE_H_

#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// Interface-specific fields.
struct InterfaceFields {
  // The following members are set at the `{` of the interface definition.

  // The interface scope.
  NameScopeId scope_id = NameScopeId::Invalid;
  // The first block of the interface body.
  // TODO: Handle control flow in the interface body, such as if-expressions.
  InstBlockId body_block_id = InstBlockId::Invalid;
  // The implicit `Self` parameter. This is a BindSymbolicName instruction.
  InstId self_param_id = InstId::Invalid;
  // The generic portion of the interface parameterized by `Self`. This always
  // has one more parameter than the interface's `generic_id`, or has exactly
  // one parameter if the interface has an invalid `generic_id`.
  GenericId generic_with_self_id = GenericId::Invalid;

  // The following members are set at the `}` of the interface definition.
  InstBlockId associated_entities_id = InstBlockId::Invalid;
};

// An interface. See EntityWithParamsBase regarding the inheritance here.
struct Interface : public EntityWithParamsBase,
                   public InterfaceFields,
                   public Printable<Interface> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{";
    PrintBaseFields(out);
    out << "}";
  }

  // Determines whether this interface has been fully defined. This is false
  // until we reach the `}` of the interface definition.
  auto is_defined() const -> bool { return associated_entities_id.is_valid(); }

  // Determines whether we're currently defining the interface. This is true
  // between the braces of the interface.
  auto is_being_defined() const -> bool {
    return definition_id.is_valid() && !is_defined();
  }
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_INTERFACE_H_
