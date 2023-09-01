// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/impl_scope.h"

#include "explorer/ast/value.h"
#include "explorer/interpreter/type_checker.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

using llvm::cast;
using llvm::dyn_cast;

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
                    const TypeChecker& type_checker,
                    std::optional<TypeStructureSortKey> sort_key) {
  if (const auto* constraint = dyn_cast<ConstraintType>(iface)) {
    CARBON_CHECK(!sort_key)
        << "should only be given a sort key for an impl of an interface";
    // The caller should have substituted `.Self` for `type` already.
    Add(constraint->impls_constraints(), deduced, impl_bindings, witness,
        type_checker);
    // A parameterized impl declaration doesn't contribute any equality
    // constraints to the scope. Instead, we'll resolve the equality
    // constraints by resolving a witness when needed.
    if (deduced.empty()) {
      for (const auto& equality_constraint :
           constraint->equality_constraints()) {
        equalities_.push_back(&equality_constraint);
      }
    }
    return;
  }

  ImplFact new_impl = {.interface = cast<InterfaceType>(iface),
                       .deduced = deduced,
                       .type = type,
                       .impl_bindings = impl_bindings,
                       .witness = witness,
                       .sort_key = std::move(sort_key)};

  // Find the first impl that's more specific than this one, and place this
  // impl right before it. This keeps the impls with the same type structure
  // sorted in lexical order, which is important for `match_first` semantics.
  auto insert_pos =
      std::upper_bound(impl_facts_.begin(), impl_facts_.end(), new_impl,
                       [](const ImplFact& a, const ImplFact& b) {
                         return a.sort_key < b.sort_key;
                       });

  impl_facts_.insert(insert_pos, std::move(new_impl));
}

void ImplScope::Add(llvm::ArrayRef<ImplsConstraint> impls_constraints,
                    llvm::ArrayRef<Nonnull<const GenericBinding*>> deduced,
                    llvm::ArrayRef<Nonnull<const ImplBinding*>> impl_bindings,
                    Nonnull<const Witness*> witness,
                    const TypeChecker& type_checker) {
  for (size_t i = 0; i != impls_constraints.size(); ++i) {
    ImplsConstraint impl = impls_constraints[i];
    Add(impl.interface, deduced, impl.type, impl_bindings,
        type_checker.MakeConstraintWitnessAccess(witness, i), type_checker);
  }
}

// Diagnose that `a_evaluated != b_evaluated` for the purpose of an equality
// constraint.
static auto DiagnoseUnequalValues(SourceLocation source_loc,
                                  Nonnull<const Value*> a_written,
                                  Nonnull<const Value*> a_evaluated,
                                  Nonnull<const Value*> b_written,
                                  Nonnull<const Value*> b_evaluated,
                                  Nonnull<const EqualityContext*> equality_ctx)
    -> Error {
  CARBON_CHECK(!ValueEqual(a_evaluated, b_evaluated, equality_ctx))
      << "expected unequal values";
  auto error = ProgramError(source_loc);
  error << "constraint requires that " << *a_written;
  if (!ValueEqual(a_written, a_evaluated, std::nullopt)) {
    error << " (with value " << *a_evaluated << ")";
  }
  error << " == " << *b_written;
  if (!ValueEqual(b_written, b_evaluated, std::nullopt)) {
    error << " (with value " << *b_evaluated << ")";
  }
  error << ", which is not known to be true";
  return std::move(error);
}

auto ImplScope::Resolve(Nonnull<const Value*> constraint_type,
                        Nonnull<const Value*> impl_type,
                        SourceLocation source_loc,
                        const TypeChecker& type_checker,
                        const Bindings& bindings) const
    -> ErrorOr<Nonnull<const Witness*>> {
  CARBON_ASSIGN_OR_RETURN(
      std::optional<Nonnull<const Witness*>> witness,
      TryResolve(constraint_type, impl_type, source_loc, type_checker, bindings,
                 /*diagnose_missing_impl=*/true));
  CARBON_CHECK(witness) << "should have diagnosed missing impl";
  return *witness;
}

auto ImplScope::TryResolve(Nonnull<const Value*> constraint_type,
                           Nonnull<const Value*> impl_type,
                           SourceLocation source_loc,
                           const TypeChecker& type_checker,
                           const Bindings& bindings,
                           bool diagnose_missing_impl) const
    -> ErrorOr<std::optional<Nonnull<const Witness*>>> {
  if (const auto* iface_type = dyn_cast<InterfaceType>(constraint_type)) {
    CARBON_ASSIGN_OR_RETURN(
        iface_type,
        type_checker.SubstituteCast<InterfaceType>(bindings, iface_type));
    return TryResolveInterface(iface_type, impl_type, source_loc, type_checker,
                               diagnose_missing_impl);
  }
  if (const auto* constraint = dyn_cast<ConstraintType>(constraint_type)) {
    std::vector<Nonnull<const Witness*>> witnesses;
    for (auto impl : constraint->impls_constraints()) {
      // Note that later impls constraints can refer to earlier impls
      // constraints via impl bindings. For example, in
      //   `C where .Self.AssocType impls D`,
      // ... the `.Self.AssocType impls D` constraint refers to the
      // `.Self impls C` constraint when naming `AssocType`. So incrementally
      // build up a partial constraint witness as we go.
      std::optional<Nonnull<const Witness*>> witness;
      if (constraint->self_binding()->impl_binding()) {
        // Note, this is a partial impl binding covering only the impl
        // constraints that we've already seen. Earlier impls constraints should
        // not be able to refer to impl bindings for later impls constraints.
        witness = type_checker.MakeConstraintWitness(witnesses);
      }
      Bindings local_bindings = bindings;
      local_bindings.Add(constraint->self_binding(), impl_type, witness);

      CARBON_ASSIGN_OR_RETURN(const auto* subst_interface,
                              type_checker.SubstituteCast<InterfaceType>(
                                  local_bindings, impl.interface));
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> subst_type,
          type_checker.Substitute(local_bindings, impl.type));
      CARBON_ASSIGN_OR_RETURN(
          std::optional<Nonnull<const Witness*>> result,
          TryResolveInterface(subst_interface, subst_type, source_loc,
                              type_checker, diagnose_missing_impl));
      if (!result) {
        return {std::nullopt};
      }
      witnesses.push_back(*result);
    }

    // Check that all intrinsic, equality, and rewrite constraints
    // are satisfied in this scope.
    llvm::ArrayRef<IntrinsicConstraint> intrinsics =
        constraint->intrinsic_constraints();
    llvm::ArrayRef<EqualityConstraint> equals =
        constraint->equality_constraints();
    llvm::ArrayRef<RewriteConstraint> rewrites =
        constraint->rewrite_constraints();
    if (!intrinsics.empty() || !equals.empty() || !rewrites.empty()) {
      std::optional<Nonnull<const Witness*>> witness;
      if (constraint->self_binding()->impl_binding()) {
        witness = type_checker.MakeConstraintWitness(witnesses);
      }
      Bindings local_bindings = bindings;
      local_bindings.Add(constraint->self_binding(), impl_type, witness);
      SingleStepEqualityContext equality_ctx(this);
      for (const auto& intrinsic : intrinsics) {
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> type,
            type_checker.Substitute(local_bindings, intrinsic.type));
        IntrinsicConstraint converted(type, intrinsic.kind, {});
        converted.arguments.reserve(intrinsic.arguments.size());
        for (Nonnull<const Value*> argument : intrinsic.arguments) {
          CARBON_ASSIGN_OR_RETURN(
              Nonnull<const Value*> subst_arg,
              type_checker.Substitute(local_bindings, argument));
          converted.arguments.push_back(subst_arg);
        }
        CARBON_ASSIGN_OR_RETURN(bool intrinsic_satisfied,
                                type_checker.IsIntrinsicConstraintSatisfied(
                                    source_loc, converted, *this));
        if (!intrinsic_satisfied) {
          if (!diagnose_missing_impl) {
            return {std::nullopt};
          }
          return ProgramError(source_loc)
                 << "constraint requires that " << converted;
        }
      }
      for (const auto& equal : equals) {
        auto it = equal.values.begin();
        CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> first,
                                type_checker.Substitute(local_bindings, *it++));
        for (; it != equal.values.end(); ++it) {
          CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> current,
                                  type_checker.Substitute(local_bindings, *it));
          if (!ValueEqual(first, current, &equality_ctx)) {
            if (!diagnose_missing_impl) {
              return {std::nullopt};
            }
            return DiagnoseUnequalValues(source_loc, equal.values.front(),
                                         first, *it, current, &equality_ctx);
          }
        }
      }
      for (const auto& rewrite : rewrites) {
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> constant,
            type_checker.Substitute(local_bindings, rewrite.constant));
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> value,
            type_checker.Substitute(local_bindings,
                                    rewrite.converted_replacement));
        if (!ValueEqual(constant, value, &equality_ctx)) {
          if (!diagnose_missing_impl) {
            return {std::nullopt};
          }
          return DiagnoseUnequalValues(source_loc, rewrite.constant, constant,
                                       rewrite.converted_replacement, value,
                                       &equality_ctx);
        }
      }
    }
    return {type_checker.MakeConstraintWitness(std::move(witnesses))};
  }
  CARBON_FATAL() << "expected a constraint, not " << *constraint_type;
}

auto ImplScope::VisitEqualValues(
    Nonnull<const Value*> value,
    llvm::function_ref<bool(Nonnull<const Value*>)> visitor) const -> bool {
  for (Nonnull<const EqualityConstraint*> eq : equalities_) {
    if (!eq->VisitEqualValues(value, visitor)) {
      return false;
    }
  }
  return !parent_scope_ || (*parent_scope_)->VisitEqualValues(value, visitor);
}

auto ImplScope::TryResolveInterface(Nonnull<const InterfaceType*> iface_type,
                                    Nonnull<const Value*> type,
                                    SourceLocation source_loc,
                                    const TypeChecker& type_checker,
                                    bool diagnose_missing_impl) const
    -> ErrorOr<std::optional<Nonnull<const Witness*>>> {
  CARBON_ASSIGN_OR_RETURN(
      std::optional<ResolveResult> result,
      TryResolveInterfaceRecursively(iface_type, type, source_loc, *this,
                                     type_checker));
  if (!result.has_value() && diagnose_missing_impl) {
    return ProgramError(source_loc) << "could not find implementation of "
                                    << *iface_type << " for " << *type;
  }
  return result ? std::optional(result->witness) : std::nullopt;
}

// Do these two witnesses refer to `impl` declarations in the same
// `match_first` block?
static auto InSameMatchFirst(Nonnull<const Witness*> a,
                             Nonnull<const Witness*> b) -> bool {
  const auto* impl_a = dyn_cast<ImplWitness>(a);
  const auto* impl_b = dyn_cast<ImplWitness>(b);
  if (!impl_a || !impl_b) {
    return false;
  }

  // TODO: Once we support an impl being declared more than once, we will need
  // to check this more carefully.
  return impl_a->declaration().match_first() &&
         impl_a->declaration().match_first() ==
             impl_b->declaration().match_first();
}

// Determine whether this result is definitely right -- that there can be no
// specialization that would give a better match.
static auto IsEffectivelyFinal(ImplScope::ResolveResult result) -> bool {
  // TODO: Once we support 'final', check whether this is a final impl
  // declaration if it's parameterized.
  return result.impl->deduced.empty();
}

// Combines the results of two impl lookups. In the event of a tie, arbitrarily
// prefer `a` over `b`.
static auto CombineResults(Nonnull<const InterfaceType*> iface_type,
                           Nonnull<const Value*> type,
                           SourceLocation source_loc,
                           std::optional<ImplScope::ResolveResult> a,
                           std::optional<ImplScope::ResolveResult> b)
    -> ErrorOr<std::optional<ImplScope::ResolveResult>> {
  // If only one lookup succeeded, return that.
  if (!b) {
    return a;
  }
  if (!a) {
    return b;
  }

  // If exactly one of them is effectively final, prefer that result.
  bool a_is_final = IsEffectivelyFinal(*a);
  bool b_is_final = IsEffectivelyFinal(*b);
  if (a_is_final && !b_is_final) {
    return a;
  } else if (b_is_final && !a_is_final) {
    return b;
  }

  const auto* impl_a = dyn_cast<ImplWitness>(a->witness);
  const auto* impl_b = dyn_cast<ImplWitness>(b->witness);

  // If both are effectively final, prefer an impl declaration over a
  // symbolic ImplBinding, because we get more information from the impl
  // declaration. If they're both symbolic, arbitrarily pick a.
  if (a_is_final && b_is_final) {
    if (!impl_b) {
      return a;
    }
    if (!impl_a) {
      return b;
    }
  }
  CARBON_CHECK(impl_a && impl_b) << "non-final impl should not be symbolic";

  // At this point, we're comparing two `impl` declarations, and either they're
  // both final or neither of them is.
  // TODO: We should reject the case where both are final when checking their
  // declarations, but we don't do so yet, so for now we report it as an
  // ambiguity.
  //
  // If they refer to the same `impl` declaration, it doesn't matter which one
  // we pick, so we pick `a`.
  // TODO: Compare the identities of the `impl`s, not the declarations.
  if (&impl_a->declaration() == &impl_b->declaration()) {
    return a;
  }

  return ProgramError(source_loc)
         << "ambiguous implementations of " << *iface_type << " for " << *type;
}

auto ImplScope::TryResolveInterfaceRecursively(
    Nonnull<const InterfaceType*> iface_type, Nonnull<const Value*> type,
    SourceLocation source_loc, const ImplScope& original_scope,
    const TypeChecker& type_checker) const
    -> ErrorOr<std::optional<ResolveResult>> {
  CARBON_ASSIGN_OR_RETURN(
      std::optional<ResolveResult> result,
      TryResolveInterfaceHere(iface_type, type, source_loc, original_scope,
                              type_checker));
  if (parent_scope_) {
    CARBON_ASSIGN_OR_RETURN(
        std::optional<ResolveResult> parent_result,
        (*parent_scope_)
            ->TryResolveInterfaceRecursively(iface_type, type, source_loc,
                                             original_scope, type_checker));
    CARBON_ASSIGN_OR_RETURN(result, CombineResults(iface_type, type, source_loc,
                                                   result, parent_result));
  }
  return result;
}

auto ImplScope::TryResolveInterfaceHere(
    Nonnull<const InterfaceType*> iface_type, Nonnull<const Value*> impl_type,
    SourceLocation source_loc, const ImplScope& original_scope,
    const TypeChecker& type_checker) const
    -> ErrorOr<std::optional<ResolveResult>> {
  std::optional<ResolveResult> result = std::nullopt;
  for (const ImplFact& impl : impl_facts_) {
    // If we've passed the final impl with a sort key matching our best impl,
    // all further are worse and don't need to be checked.
    if (result && result->impl->sort_key < impl.sort_key) {
      break;
    }

    // If this impl appears later in the same match_first block as our best
    // result, we should not consider it.
    //
    // TODO: This should apply transitively: if we have
    //   match_first { impl a; impl b; }
    //   match_first { impl b; impl c; }
    // then we should not consider c once we match a. For now, because each
    // impl is only declared once, this is not a problem.
    if (result && InSameMatchFirst(result->impl->witness, impl.witness)) {
      continue;
    }

    // Try matching this impl against our query.
    CARBON_ASSIGN_OR_RETURN(std::optional<Nonnull<const Witness*>> witness,
                            type_checker.MatchImpl(*iface_type, impl_type, impl,
                                                   original_scope, source_loc));
    if (witness) {
      CARBON_ASSIGN_OR_RETURN(
          result,
          CombineResults(iface_type, impl_type, source_loc, result,
                         ResolveResult{.impl = &impl, .witness = *witness}));
    }
  }
  return result;
}

// TODO: Add indentation when printing the parents.
void ImplScope::Print(llvm::raw_ostream& out) const {
  llvm::ListSeparator sep(",\n    ");
  out << "    "
      << "[";
  for (const ImplFact& impl : impl_facts_) {
    out << sep << "`" << *(impl.type) << "` as `" << *(impl.interface) << "`";
    if (impl.sort_key) {
      out << " " << *impl.sort_key;
    }
  }
  for (Nonnull<const EqualityConstraint*> eq : equalities_) {
    out << sep;
    llvm::ListSeparator equal(" == ");
    for (Nonnull<const Value*> value : eq->values) {
      out << equal << "`" << *value << "`";
    }
  }
  out << "]\n";
  if (parent_scope_) {
    out << **parent_scope_;
  }
}

auto SingleStepEqualityContext::VisitEqualValues(
    Nonnull<const Value*> value,
    llvm::function_ref<bool(Nonnull<const Value*>)> visitor) const -> bool {
  return impl_scope_->VisitEqualValues(value, visitor);
}

}  // namespace Carbon
