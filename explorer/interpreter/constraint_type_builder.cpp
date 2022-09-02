// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/constraint_type_builder.h"

namespace Carbon {

ConstraintTypeBuilder::ConstraintTypeBuilder(Nonnull<Arena*> arena,
                                             SourceLocation source_loc)
    : ConstraintTypeBuilder(MakeSelfBinding(arena, source_loc)) {}

ConstraintTypeBuilder::ConstraintTypeBuilder(
    Nonnull<const GenericBinding*> self_binding)
    : self_binding_(self_binding) {}

void ConstraintTypeBuilder::AddImplConstraint(
    ConstraintType::ImplConstraint impl) {
  for (ConstraintType::ImplConstraint existing : impl_constraints_) {
    if (TypeEqual(existing.type, impl.type, std::nullopt) &&
        TypeEqual(existing.interface, impl.interface, std::nullopt)) {
      return;
    }
  }
  impl_constraints_.push_back(std::move(impl));
}

void ConstraintTypeBuilder::AddEqualityConstraint(
    ConstraintType::EqualityConstraint equal) {
  CARBON_CHECK(equal.values.size() >= 2) << "degenerate equality constraint";

  // TODO: Check to see if this constraint is already present and deduplicate
  // if so. We could also look for a superset / subset and keep the larger
  // one. We could in theory detect `A == B and B == C and C == A` and merge
  // into a single `A == B == C` constraint, but that's more work than it's
  // worth doing here.
  equality_constraints_.push_back(std::move(equal));
}

void ConstraintTypeBuilder::AddLookupContext(
    ConstraintType::LookupContext context) {
  for (ConstraintType::LookupContext existing : lookup_contexts_) {
    if (ValueEqual(existing.context, context.context, std::nullopt)) {
      return;
    }
  }
  lookup_contexts_.push_back(std::move(context));
}

void ConstraintTypeBuilder::Add(Nonnull<const ConstraintType*> constraint) {
  for (const auto& impl_constraint : constraint->impl_constraints()) {
    AddImplConstraint(impl_constraint);
  }

  for (const auto& equality_constraint : constraint->equality_constraints()) {
    AddEqualityConstraint(equality_constraint);
  }

  for (const auto& lookup_context : constraint->lookup_contexts()) {
    AddLookupContext(lookup_context);
  }
}

auto ConstraintTypeBuilder::Build(
    Nonnull<Arena*> arena_) && -> Nonnull<const ConstraintType*> {
  return arena_->New<ConstraintType>(
      self_binding_, std::move(impl_constraints_),
      std::move(equality_constraints_), std::move(lookup_contexts_));
}

// static
auto ConstraintTypeBuilder::MakeSelfBinding(Nonnull<Arena*> arena,
                                            SourceLocation source_loc)
    -> Nonnull<const GenericBinding*> {
  Nonnull<GenericBinding*> self_binding = arena->New<GenericBinding>(
      source_loc, ".Self", arena->New<TypeTypeLiteral>(source_loc));
  Nonnull<const Value*> self = arena->New<VariableType>(self_binding);
  // TODO: Do we really need both of these?
  self_binding->set_symbolic_identity(self);
  self_binding->set_value(self);
  return self_binding;
}

}  // namespace Carbon
