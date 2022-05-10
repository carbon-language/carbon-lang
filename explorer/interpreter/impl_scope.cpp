// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/impl_scope.h"

#include "explorer/common/error.h"
#include "explorer/interpreter/value.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

using llvm::cast;

namespace Carbon {

void ImplScope::Add(Nonnull<const Value*> iface, Nonnull<const Value*> type,
                    ValueNodeView impl) {
  impls_.push_back({.interface = iface, .type = type, .impl = impl});
}

void ImplScope::AddParent(Nonnull<const ImplScope*> parent) {
  parent_scopes_.push_back(parent);
}

auto ImplScope::Resolve(Nonnull<const Value*> iface_type,
                        Nonnull<const Value*> type,
                        SourceLocation source_loc) const
    -> ErrorOr<ValueNodeView> {
  ASSIGN_OR_RETURN(std::optional<ValueNodeView> result,
                   TryResolve(iface_type, type, source_loc));
  if (!result.has_value()) {
    return FATAL_COMPILATION_ERROR(source_loc)
           << "could not find implementation of " << *iface_type << " for "
           << *type;
  }
  return *result;
}

auto ImplScope::TryResolve(Nonnull<const Value*> iface_type,
                           Nonnull<const Value*> type,
                           SourceLocation source_loc) const
    -> ErrorOr<std::optional<ValueNodeView>> {
  std::optional<ValueNodeView> result =
      ResolveHere(iface_type, type, source_loc);
  if (result.has_value()) {
    return result;
  }
  for (Nonnull<const ImplScope*> parent : parent_scopes_) {
    ASSIGN_OR_RETURN(auto parent_result,
                     parent->TryResolve(iface_type, type, source_loc));
    if (parent_result.has_value() && result.has_value() &&
        *parent_result != *result) {
      return FATAL_COMPILATION_ERROR(source_loc)
             << "ambiguous implementations of " << *iface_type << " for "
             << *type;
    }
    result = parent_result;
  }
  return result;
}

auto ImplScope::ResolveHere(Nonnull<const Value*> iface_type,
                            Nonnull<const Value*> impl_type,
                            SourceLocation /*source_loc*/) const
    -> std::optional<ValueNodeView> {
  switch (iface_type->kind()) {
    case Value::Kind::InterfaceType: {
      const auto& iface = cast<InterfaceType>(*iface_type);
      for (const Impl& impl : impls_) {
        if (TypeEqual(&iface, impl.interface) &&
            TypeEqual(impl_type, impl.type)) {
          return impl.impl;
        }
      }
      return std::nullopt;
    }
    default:
      FATAL() << "expected an interface, not " << *iface_type;
      break;
  }
}

// TODO: Add indentation when printing the parents.
void ImplScope::Print(llvm::raw_ostream& out) const {
  out << "impls: ";
  llvm::ListSeparator sep;
  for (Impl impl : impls_) {
    out << sep << *(impl.type) << " as " << *(impl.interface);
  }
  out << "\n";
  for (const Nonnull<const ImplScope*>& parent : parent_scopes_) {
    out << *parent;
  }
}

}  // namespace Carbon
