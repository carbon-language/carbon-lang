// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_CONSTRAINT_TYPE_BUILDER_H_
#define CARBON_EXPLORER_INTERPRETER_CONSTRAINT_TYPE_BUILDER_H_

#include "explorer/ast/pattern.h"
#include "explorer/common/arena.h"
#include "explorer/common/source_location.h"
#include "explorer/interpreter/value.h"

namespace Carbon {

// Builder for constraint types.
//
// This type supports incrementally building a constraint type by adding
// constraints one at a time, and will deduplicate the constraints as it goes.
//
// TODO: The deduplication here is very inefficient. We should use value
// canonicalization or hashing or similar to speed this up.
class ConstraintTypeBuilder {
 public:
  ConstraintTypeBuilder(Nonnull<Arena*> arena, SourceLocation source_loc);

  explicit ConstraintTypeBuilder(Nonnull<const GenericBinding*> self_binding);

  // Produce a type that refers to the `.Self` type of the constraint.
  auto GetSelfType(Nonnull<Arena*> arena) const -> Nonnull<const Value*> {
    return &self_binding_->value();
  }

  // Add an `impl` constraint -- `T is C` if not already present.
  void AddImplConstraint(ConstraintType::ImplConstraint impl);

  // Add an equality constraint -- `A == B`.
  void AddEqualityConstraint(ConstraintType::EqualityConstraint equal);

  // Add a context for qualified name lookup, if not already present.
  void AddLookupContext(ConstraintType::LookupContext context);

  // Add all the constraints from another constraint type. The constraints must
  // not refer to that other constraint type's self binding, because it will no
  // longer be in scope.
  void Add(Nonnull<const ConstraintType*> constraint);

  // Convert the builder into a ConstraintType. Note that this consumes the
  // builder.
  auto Build(Nonnull<Arena*> arena) && -> Nonnull<const ConstraintType*>;

 private:
  // Make a generic binding to serve as the `.Self` of this constraint type.
  static auto MakeSelfBinding(Nonnull<Arena*> arena, SourceLocation source_loc)
      -> Nonnull<const GenericBinding*>;

  Nonnull<const GenericBinding*> self_binding_;
  std::vector<ConstraintType::ImplConstraint> impl_constraints_;
  std::vector<ConstraintType::EqualityConstraint> equality_constraints_;
  std::vector<ConstraintType::LookupContext> lookup_contexts_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_CONSTRAINT_TYPE_BUILDER_H_
