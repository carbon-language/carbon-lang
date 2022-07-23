// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/impl_scope.h"

#include "explorer/interpreter/type_checker.h"
#include "explorer/interpreter/value.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

using llvm::cast;
using llvm::dyn_cast;

namespace Carbon {

void ImplScope::Add(Nonnull<const Value*> iface, Nonnull<const Value*> type,
                    Nonnull<Expression*> impl,
                    const TypeChecker& type_checker) {
  Add(iface, {}, type, {}, impl, type_checker);
}

void ImplScope::Add(Nonnull<const Value*> iface,
                    llvm::ArrayRef<Nonnull<const GenericBinding*>> deduced,
                    Nonnull<const Value*> type,
                    llvm::ArrayRef<Nonnull<const ImplBinding*>> impl_bindings,
                    Nonnull<Expression*> impl_expr,
                    const TypeChecker& type_checker) {
  if (auto* orig_constraint = dyn_cast<ConstraintType>(iface)) {
    BindingMap map;
    map[orig_constraint->self_binding()] = type;
    const ConstraintType* constraint =
        cast<ConstraintType>(type_checker.Substitute(map, orig_constraint));
    for (size_t i = 0; i != constraint->impl_constraints().size(); ++i) {
      ConstraintType::ImplConstraint impl = constraint->impl_constraints()[i];
      Add(impl.interface, deduced, impl.type, impl_bindings,
          type_checker.MakeConstraintWitnessAccess(impl_expr, i), type_checker);
    }
    // A parameterized impl declaration doesn't contribute any equality
    // constraints to the scope. Instead, we'll resolve the equality
    // constraints by resolving a witness when needed.
    if (deduced.empty()) {
      for (auto& equality_constraint : constraint->equality_constraints()) {
        equalities_.push_back(&equality_constraint);
      }
    }
    return;
  }

  impls_.push_back({.interface = cast<InterfaceType>(iface),
                    .deduced = deduced,
                    .type = type,
                    .impl_bindings = impl_bindings,
                    .impl = impl_expr});
}

void ImplScope::AddParent(Nonnull<const ImplScope*> parent) {
  parent_scopes_.push_back(parent);
}

auto ImplScope::Resolve(Nonnull<const Value*> constraint_type,
                        Nonnull<const Value*> impl_type,
                        SourceLocation source_loc,
                        const TypeChecker& type_checker) const
    -> ErrorOr<Nonnull<Expression*>> {
  if (const auto* iface_type = dyn_cast<InterfaceType>(constraint_type)) {
    return ResolveInterface(iface_type, impl_type, source_loc, type_checker);
  }
  if (const auto* constraint = dyn_cast<ConstraintType>(constraint_type)) {
    std::vector<Nonnull<Expression*>> witnesses;
    BindingMap map;
    map[constraint->self_binding()] = impl_type;
    for (auto impl : constraint->impl_constraints()) {
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<Expression*> result,
          ResolveInterface(
              cast<InterfaceType>(type_checker.Substitute(map, impl.interface)),
              type_checker.Substitute(map, impl.type), source_loc,
              type_checker));
      witnesses.push_back(result);
    }
    // TODO: Check satisfaction of same-type constraints.
    return type_checker.MakeConstraintWitness(*constraint, std::move(witnesses),
                                              source_loc);
  }
  CARBON_FATAL() << "expected a constraint, not " << *constraint_type;
}

auto ImplScope::VisitEqualValues(
    Nonnull<const Value*> value,
    llvm::function_ref<bool(Nonnull<const Value*>)> visitor) const -> bool {
  for (Nonnull<const ConstraintType::EqualityConstraint*> eq : equalities_) {
    if (!eq->VisitEqualValues(value, visitor)) {
      return false;
    }
  }
  for (Nonnull<const ImplScope*> parent : parent_scopes_) {
    if (!parent->VisitEqualValues(value, visitor)) {
      return false;
    }
  }
  return true;
}

auto ImplScope::ResolveInterface(Nonnull<const InterfaceType*> iface_type,
                                 Nonnull<const Value*> type,
                                 SourceLocation source_loc,
                                 const TypeChecker& type_checker) const
    -> ErrorOr<Nonnull<Expression*>> {
  CARBON_ASSIGN_OR_RETURN(
      std::optional<Nonnull<Expression*>> result,
      TryResolve(iface_type, type, source_loc, *this, type_checker));
  if (!result.has_value()) {
    return CompilationError(source_loc) << "could not find implementation of "
                                        << *iface_type << " for " << *type;
  }
  return *result;
}

auto ImplScope::TryResolve(Nonnull<const InterfaceType*> iface_type,
                           Nonnull<const Value*> type,
                           SourceLocation source_loc,
                           const ImplScope& original_scope,
                           const TypeChecker& type_checker) const
    -> ErrorOr<std::optional<Nonnull<Expression*>>> {
  CARBON_ASSIGN_OR_RETURN(
      std::optional<Nonnull<Expression*>> result,
      ResolveHere(iface_type, type, source_loc, original_scope, type_checker));
  for (Nonnull<const ImplScope*> parent : parent_scopes_) {
    CARBON_ASSIGN_OR_RETURN(std::optional<Nonnull<Expression*>> parent_result,
                            parent->TryResolve(iface_type, type, source_loc,
                                               original_scope, type_checker));
    if (parent_result.has_value()) {
      if (result.has_value()) {
        return CompilationError(source_loc) << "ambiguous implementations of "
                                            << *iface_type << " for " << *type;
      } else {
        result = *parent_result;
      }
    }
  }
  return result;
}

auto ImplScope::ResolveHere(Nonnull<const InterfaceType*> iface_type,
                            Nonnull<const Value*> impl_type,
                            SourceLocation source_loc,
                            const ImplScope& original_scope,
                            const TypeChecker& type_checker) const
    -> ErrorOr<std::optional<Nonnull<Expression*>>> {
  std::optional<Nonnull<Expression*>> result = std::nullopt;
  for (const Impl& impl : impls_) {
    std::optional<Nonnull<Expression*>> m = type_checker.MatchImpl(
        *iface_type, impl_type, impl, original_scope, source_loc);
    if (m.has_value()) {
      if (result.has_value()) {
        return CompilationError(source_loc)
               << "ambiguous implementations of " << *iface_type << " for "
               << *impl_type;
      } else {
        result = *m;
      }
    }
  }
  return result;
}

// TODO: Add indentation when printing the parents.
void ImplScope::Print(llvm::raw_ostream& out) const {
  out << "impls: ";
  llvm::ListSeparator sep;
  for (const Impl& impl : impls_) {
    out << sep << *(impl.type) << " as " << *(impl.interface);
  }
  for (Nonnull<const ConstraintType::EqualityConstraint*> eq : equalities_) {
    out << sep;
    llvm::ListSeparator equal(" == ");
    for (Nonnull<const Value*> value : eq->values) {
      out << equal << *value;
    }
  }
  out << "\n";
  for (const Nonnull<const ImplScope*>& parent : parent_scopes_) {
    out << *parent;
  }
}

}  // namespace Carbon
