// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/impl_scope.h"

#include "explorer/interpreter/type_checker.h"
#include "explorer/interpreter/value.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

using llvm::cast;

namespace Carbon {

void ImplScope::Add(Nonnull<const Value*> iface, Nonnull<const Value*> type,
                    Nonnull<Expression*> impl) {
  Add(iface, {}, type, {}, impl);
}

void ImplScope::Add(Nonnull<const Value*> iface,
                    llvm::ArrayRef<Nonnull<const GenericBinding*>> deduced,
                    Nonnull<const Value*> type,
                    llvm::ArrayRef<Nonnull<const ImplBinding*>> impl_bindings,
                    Nonnull<Expression*> impl) {
  impls_.push_back({.interface = iface,
                    .deduced = deduced,
                    .type = type,
                    .impl_bindings = impl_bindings,
                    .impl = impl});
}

void ImplScope::AddParent(Nonnull<const ImplScope*> parent) {
  parent_scopes_.push_back(parent);
}

auto ImplScope::Resolve(Nonnull<const Value*> iface_type,
                        Nonnull<const Value*> type, SourceLocation source_loc,
                        const TypeChecker& type_checker) const
    -> ErrorOr<Nonnull<Expression*>> {
  CARBON_ASSIGN_OR_RETURN(
      std::optional<ImplResult> result,
      TryResolve(iface_type, type, source_loc, *this, type_checker));
  if (!result.has_value()) {
    return CompilationError(source_loc) << "could not find implementation of "
                                        << *iface_type << " for " << *type;
  }
  return (*result).impl_expression;
}

auto ImplScope::SelectImpl(const ImplScope::ImplResult& impl1,
                           const ImplScope::ImplResult& impl2,
                           SourceLocation source_loc,
                           const ImplScope& original_scope,
                           const TypeChecker& type_checker,
                           Nonnull<const Value*> iface_type,
                           Nonnull<const Value*> impl_type) const
    -> ErrorOr<std::optional<ImplScope::ImplResult>> {
  std::optional<Nonnull<Expression*>> result1 = type_checker.MatchImpl(
      *cast<InterfaceType>(impl1.impl.interface), impl1.impl.type, impl2.impl,
      original_scope, source_loc);
  std::optional<Nonnull<Expression*>> result2 = type_checker.MatchImpl(
      *cast<InterfaceType>(impl2.impl.interface), impl2.impl.type, impl1.impl,
      original_scope, source_loc);

  if (result1.has_value() && result2.has_value()) {
    return CompilationError(source_loc) << "ambiguous implementations of "
                                        << *iface_type << " for " << *impl_type;
  } else if (result1.has_value()) {
    std::optional<ImplScope::ImplResult> result1 = impl1;
    return result1;
  } else {
    std::optional<ImplScope::ImplResult> result2 = impl2;
    return result2;
  }
}

auto ImplScope::TryResolve(Nonnull<const Value*> iface_type,
                           Nonnull<const Value*> type,
                           SourceLocation source_loc,
                           const ImplScope& original_scope,
                           const TypeChecker& type_checker) const
    -> ErrorOr<std::optional<ImplResult>> {
  CARBON_ASSIGN_OR_RETURN(
      std::optional<ImplResult> result,
      ResolveHere(iface_type, type, source_loc, original_scope, type_checker));
  for (Nonnull<const ImplScope*> parent : parent_scopes_) {
    CARBON_ASSIGN_OR_RETURN(std::optional<ImplResult> parent_result,
                            parent->TryResolve(iface_type, type, source_loc,
                                               original_scope, type_checker));
    if (parent_result.has_value()) {
      if (result.has_value()) {
        return SelectImpl(*parent_result, *result, source_loc, original_scope,
                          type_checker, iface_type, type);
      } else {
        result = *parent_result;
      }
    }
  }
  return result;
}

auto ImplScope::ResolveHere(Nonnull<const Value*> iface_type,
                            Nonnull<const Value*> impl_type,
                            SourceLocation source_loc,
                            const ImplScope& original_scope,
                            const TypeChecker& type_checker) const
    -> ErrorOr<std::optional<ImplResult>> {
  if (iface_type->kind() != Value::Kind::InterfaceType) {
    CARBON_FATAL() << "expected an interface, not " << *iface_type;
  }
  const auto& iface = cast<InterfaceType>(*iface_type);
  std::optional<ImplResult> result = std::nullopt;
  for (const Impl& impl : impls_) {
    std::optional<Nonnull<Expression*>> m = type_checker.MatchImpl(
        iface, impl_type, impl, original_scope, source_loc);
    if (m.has_value()) {
      if (result.has_value()) {
        return SelectImpl(*result, {*m, impl}, source_loc, original_scope,
                          type_checker, iface_type, impl_type);
      } else {
        result = {*m, impl};
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
  out << "\n";
  for (const Nonnull<const ImplScope*>& parent : parent_scopes_) {
    out << *parent;
  }
}

}  // namespace Carbon
