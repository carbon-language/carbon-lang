// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_CLASS_H_
#define CARBON_TOOLCHAIN_SEM_IR_CLASS_H_

#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// A class.
struct Class : public Printable<Class> {
  enum InheritanceKind : int8_t {
    // `abstract class`
    Abstract,
    // `base class`
    Base,
    // `class`
    Final,
  };

  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{name: " << name_id << ", enclosing_scope: " << enclosing_scope_id
        << "}";
  }

  // Determines whether this class has been fully defined. This is false until
  // we reach the `}` of the class definition.
  auto is_defined() const -> bool { return object_repr_id.is_valid(); }

  // The following members always have values, and do not change throughout the
  // lifetime of the class.

  // The class name.
  NameId name_id;
  // The enclosing scope.
  NameScopeId enclosing_scope_id;
  // The class type, which is the type of `Self` in the class definition.
  TypeId self_type_id;
  // The first declaration of the class. This is a ClassDecl.
  InstId decl_id = InstId::Invalid;
  // The kind of inheritance that this class supports.
  // TODO: The rules here are not yet decided. See #3384.
  InheritanceKind inheritance_kind;

  // The following members are set at the `{` of the class definition.

  // The definition of the class. This is a ClassDecl.
  InstId definition_id = InstId::Invalid;
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

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_CLASS_H_
