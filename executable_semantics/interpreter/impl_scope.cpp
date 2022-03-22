// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/impl_scope.h"

#include "executable_semantics/common/error.h"
#include "executable_semantics/interpreter/value.h"
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
                        SourceLocation source_loc) const -> ValueNodeView {
  std::optional<ValueNodeView> result =
      TryResolve(iface_type, type, source_loc);
  if (!result.has_value()) {
    FATAL_COMPILATION_ERROR(source_loc) << "could not find implementation of "
                                        << *iface_type << " for " << *type;
  }
  return *result;
}

auto ImplScope::TryResolve(Nonnull<const Value*> iface_type,
                           Nonnull<const Value*> type,
                           SourceLocation source_loc) const
    -> std::optional<ValueNodeView> {
  std::optional<ValueNodeView> result =
      ResolveHere(iface_type, type, source_loc);
  if (result.has_value()) {
    return result;
  }
  for (Nonnull<const ImplScope*> parent : parent_scopes_) {
    auto parent_result = parent->TryResolve(iface_type, type, source_loc);
    if (parent_result.has_value() && result.has_value() &&
        *parent_result != *result) {
      FATAL_COMPILATION_ERROR(source_loc)
          << "ambiguous implementations of " << *iface_type << " for " << *type;
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

}  // namespace Carbon
