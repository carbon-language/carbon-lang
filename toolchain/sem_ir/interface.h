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
  NameScopeId scope_id = NameScopeId::None;
  // The first block of the interface body.
  // TODO: Handle control flow in the interface body, such as if-expressions.
  InstBlockId body_block_id = InstBlockId::None;
  // The implicit `Self` parameter. This is a BindSymbolicName instruction.
  InstId self_param_id = InstId::None;

  // The following members are set at the `}` of the interface definition.
  InstBlockId associated_entities_id = InstBlockId::None;
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

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_INTERFACE_H_
