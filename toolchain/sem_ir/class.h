// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_CLASS_H_
#define CARBON_TOOLCHAIN_SEM_IR_CLASS_H_

#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// Class-specific fields.
struct ClassFields {
  enum InheritanceKind : int8_t {
    // `abstract class`
    Abstract,
    // `base class`
    Base,
    // `class`
    Final,
  };

  // The following members always have values, and do not change throughout the
  // lifetime of the class.

  // The class type, which is the type of `Self` in the class definition.
  TypeId self_type_id;
  // The kind of inheritance that this class supports.
  // TODO: The rules here are not yet decided. See #3384.
  InheritanceKind inheritance_kind;

  // The following members are set at the `{` of the class definition.

  // The class scope.
  NameScopeId scope_id = NameScopeId::Invalid;
  // The first block of the class body.
  // TODO: Handle control flow in the class body, such as if-expressions.
  InstBlockId body_block_id = InstBlockId::Invalid;

  // The following members are accumulated throughout the class definition.

  // The adapted type declaration, if any. Invalid if the class is not an
  // adapter. This is an AdaptDecl instruction.
  // TODO: Consider sharing the storage for `adapt_id` and `base_id`. A class
  // can't have both.
  InstId adapt_id = InstId::Invalid;
  // The base class declaration. Invalid if the class has no base class. This is
  // a BaseDecl instruction.
  InstId base_id = InstId::Invalid;

  // The following members are set at the `}` of the class definition.

  // The object representation type to use for this class. This is valid once
  // the class is defined. For an adapter, this is the non-adapter type that
  // this class directly or transitively adapts.
  TypeId object_repr_id = TypeId::Invalid;
};

// A class. See EntityWithParamsBase regarding the inheritance here.
struct Class : public EntityWithParamsBase,
               public ClassFields,
               public Printable<Class> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{";
    PrintBaseFields(out);
    out << "}";
  }

  // Determines whether this class has been fully defined. This is false until
  // we reach the `}` of the class definition.
  auto is_defined() const -> bool { return object_repr_id.is_valid(); }
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_CLASS_H_
