// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/impl_scope.h"

#include "executable_semantics/interpreter/type_checker.h"
#include "executable_semantics/interpreter/value.h"
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
  ASSIGN_OR_RETURN(
      std::optional<Nonnull<Expression*>> result,
      TryResolve(iface_type, type, source_loc, *this, type_checker));
  if (!result.has_value()) {
    return CompilationError(source_loc) << "could not find implementation of "
                                        << *iface_type << " for " << *type;
  }
  return *result;
}

auto ImplScope::TryResolve(Nonnull<const Value*> iface_type,
                           Nonnull<const Value*> type,
                           SourceLocation source_loc,
                           const ImplScope& original_scope,
                           const TypeChecker& type_checker) const
    -> ErrorOr<std::optional<Nonnull<Expression*>>> {
  ASSIGN_OR_RETURN(
      std::optional<Nonnull<Expression*>> result,
      ResolveHere(iface_type, type, source_loc, original_scope, type_checker));
  for (Nonnull<const ImplScope*> parent : parent_scopes_) {
    ASSIGN_OR_RETURN(std::optional<Nonnull<Expression*>> parent_result,
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

auto ImplScope::ResolveHere(Nonnull<const Value*> iface_type,
                            Nonnull<const Value*> impl_type,
                            SourceLocation source_loc,
                            const ImplScope& original_scope,
                            const TypeChecker& type_checker) const
    -> ErrorOr<std::optional<Nonnull<Expression*>>> {
  if (iface_type->kind() != Value::Kind::InterfaceType) {
    FATAL() << "expected an interface, not " << *iface_type;
  }
  const auto& iface = cast<InterfaceType>(*iface_type);
  std::optional<Nonnull<Expression*>> result = std::nullopt;
  for (const Impl& impl : impls_) {
    std::optional<Nonnull<Expression*>> m = type_checker.MatchImpl(
        iface, impl_type, impl, original_scope, source_loc);
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
  out << "\n";
  for (const Nonnull<const ImplScope*>& parent : parent_scopes_) {
    out << *parent;
  }
}

}  // namespace Carbon
