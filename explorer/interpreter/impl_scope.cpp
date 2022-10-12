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
using llvm::isa;

namespace Carbon {

void ImplScope::Add(Nonnull<const Value*> iface, Nonnull<const Value*> type,
                    Nonnull<const Witness*> witness,
                    const TypeChecker& type_checker) {
  Add(iface, {}, type, {}, witness, type_checker);
}

void ImplScope::Add(Nonnull<const Value*> iface,
                    llvm::ArrayRef<Nonnull<const GenericBinding*>> deduced,
                    Nonnull<const Value*> type,
                    llvm::ArrayRef<Nonnull<const ImplBinding*>> impl_bindings,
                    Nonnull<const Witness*> witness,
                    const TypeChecker& type_checker) {
  if (auto* constraint = dyn_cast<ConstraintType>(iface)) {
    // The caller should have substituted `.Self` for `type` already.
    Add(constraint->impl_constraints(), deduced, impl_bindings, witness,
        type_checker);
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
                    .witness = witness});
}

void ImplScope::Add(llvm::ArrayRef<ConstraintType::ImplConstraint> impls,
                    llvm::ArrayRef<Nonnull<const GenericBinding*>> deduced,
                    llvm::ArrayRef<Nonnull<const ImplBinding*>> impl_bindings,
                    Nonnull<const Witness*> witness,
                    const TypeChecker& type_checker) {
  for (size_t i = 0; i != impls.size(); ++i) {
    ConstraintType::ImplConstraint impl = impls[i];
    Add(impl.interface, deduced, impl.type, impl_bindings,
        type_checker.MakeConstraintWitnessAccess(witness, i), type_checker);
  }
}

void ImplScope::AddParent(Nonnull<const ImplScope*> parent) {
  parent_scopes_.push_back(parent);
}

auto ImplScope::Resolve(Nonnull<const Value*> constraint_type,
                        Nonnull<const Value*> impl_type,
                        SourceLocation source_loc,
                        const TypeChecker& type_checker,
                        const Bindings& bindings) const
    -> ErrorOr<Nonnull<const Witness*>> {
  if (const auto* iface_type = dyn_cast<InterfaceType>(constraint_type)) {
    iface_type =
        cast<InterfaceType>(type_checker.Substitute(bindings, iface_type));
    return ResolveInterface(iface_type, impl_type, source_loc, type_checker);
  }
  if (const auto* constraint = dyn_cast<ConstraintType>(constraint_type)) {
    std::vector<Nonnull<const Witness*>> witnesses;
    for (auto impl : constraint->impl_constraints()) {
      // Note that later impl constraints can refer to earlier impl constraints
      // via impl bindings. For example, in
      //   `C where .Self.AssocType is D`,
      // ... the `.Self.AssocType is D` constraint refers to the `.Self is C`
      // constraint when naming `AssocType`. So incrementally build up a
      // partial constraint witness as we go.
      std::optional<Nonnull<const Witness*>> witness;
      if (constraint->self_binding()->impl_binding()) {
        // Note, this is a partial impl binding covering only the impl
        // constraints that we've already seen. Earlier impl constraints should
        // not be able to refer to impl bindings for later impl constraints.
        witness = type_checker.MakeConstraintWitness(*constraint, witnesses,
                                                     source_loc);
      }
      Bindings local_bindings = bindings;
      local_bindings.Add(constraint->self_binding(), impl_type, witness);
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Witness*> result,
          ResolveInterface(cast<InterfaceType>(type_checker.Substitute(
                               local_bindings, impl.interface)),
                           type_checker.Substitute(local_bindings, impl.type),
                           source_loc, type_checker));
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
    -> ErrorOr<Nonnull<const Witness*>> {
  CARBON_ASSIGN_OR_RETURN(
      std::optional<Nonnull<const Witness*>> result,
      TryResolve(iface_type, type, source_loc, *this, type_checker));
  if (!result.has_value()) {
    return ProgramError(source_loc) << "could not find implementation of "
                                    << *iface_type << " for " << *type;
  }
  return *result;
}

// Combines the results of two impl lookups. In the event of a tie, arbitrarily
// prefer `a` over `b`.
static auto CombineResults(Nonnull<const InterfaceType*> iface_type,
                           Nonnull<const Value*> type,
                           SourceLocation source_loc,
                           std::optional<Nonnull<const Witness*>> a,
                           std::optional<Nonnull<const Witness*>> b)
    -> ErrorOr<std::optional<Nonnull<const Witness*>>> {
  // If only one lookup succeeded, return that.
  if (!b) {
    return a;
  }
  if (!a) {
    return b;
  }
  // If either of them was a symbolic result, then they'll end up being
  // equivalent. In that case, pick `a`.
  auto* impl_a = dyn_cast<ImplWitness>(*a);
  auto* impl_b = dyn_cast<ImplWitness>(*b);
  if (!impl_b) {
    return a;
  }
  if (!impl_a) {
    return b;
  }
  // If they refer to the same `impl` declaration, it doesn't matter which one
  // we pick, so we pick `a`.
  // TODO: Compare the identities of the `impl`s, not the declarations.
  if (&impl_a->declaration() == &impl_b->declaration()) {
    return a;
  }
  // TODO: Order the `impl`s based on type structure.
  return ProgramError(source_loc)
         << "ambiguous implementations of " << *iface_type << " for " << *type;
}

auto ImplScope::TryResolve(Nonnull<const InterfaceType*> iface_type,
                           Nonnull<const Value*> type,
                           SourceLocation source_loc,
                           const ImplScope& original_scope,
                           const TypeChecker& type_checker) const
    -> ErrorOr<std::optional<Nonnull<const Witness*>>> {
  CARBON_ASSIGN_OR_RETURN(
      std::optional<Nonnull<const Witness*>> result,
      ResolveHere(iface_type, type, source_loc, original_scope, type_checker));
  for (Nonnull<const ImplScope*> parent : parent_scopes_) {
    CARBON_ASSIGN_OR_RETURN(
        std::optional<Nonnull<const Witness*>> parent_result,
        parent->TryResolve(iface_type, type, source_loc, original_scope,
                           type_checker));
    CARBON_ASSIGN_OR_RETURN(result, CombineResults(iface_type, type, source_loc,
                                                   result, parent_result));
  }
  return result;
}

auto ImplScope::ResolveHere(Nonnull<const InterfaceType*> iface_type,
                            Nonnull<const Value*> impl_type,
                            SourceLocation source_loc,
                            const ImplScope& original_scope,
                            const TypeChecker& type_checker) const
    -> ErrorOr<std::optional<Nonnull<const Witness*>>> {
  std::optional<Nonnull<const Witness*>> result = std::nullopt;
  for (const Impl& impl : impls_) {
    std::optional<Nonnull<const Witness*>> m = type_checker.MatchImpl(
        *iface_type, impl_type, impl, original_scope, source_loc);
    CARBON_ASSIGN_OR_RETURN(
        result, CombineResults(iface_type, impl_type, source_loc, result, m));
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
