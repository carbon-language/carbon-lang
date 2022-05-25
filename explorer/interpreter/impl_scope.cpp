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
  std::list<ImplResult> result;
  TryResolve(iface_type, type, source_loc, *this, type_checker, result);
  return SelectImpl(result, iface_type, type, source_loc, *this, type_checker);
}

auto ImplScope::MoreSpecificImpl(const ImplScope::ImplResult& impl1,
                                 const ImplScope::ImplResult& impl2,
                                 SourceLocation source_loc,
                                 const ImplScope& original_scope,
                                 const TypeChecker& type_checker) const
    -> ImplComparison {
  std::optional<Nonnull<Expression*>> result1 = type_checker.MatchImpl(
      *cast<InterfaceType>(impl1.impl.interface), impl1.impl.type, impl2.impl,
      original_scope, source_loc);
  std::optional<Nonnull<Expression*>> result2 = type_checker.MatchImpl(
      *cast<InterfaceType>(impl2.impl.interface), impl2.impl.type, impl1.impl,
      original_scope, source_loc);
  if (result1.has_value() && result2.has_value()) {
    return ImplComparison::EquallySpecific;
  } else if (result1.has_value() && !result2.has_value()) {
    return ImplComparison::MoreSpecific;
  } else if (!result1.has_value() && result2.has_value()) {
    return ImplComparison::LessSpecific;
  } else {
    return ImplComparison::Incomparable;
  }
}

auto ImplScope::SelectImpl(const std::list<ImplScope::ImplResult>& impls,
                           Nonnull<const Value*> iface_type,
                           Nonnull<const Value*> impl_type,
                           SourceLocation source_loc,
                           const ImplScope& original_scope,
                           const TypeChecker& type_checker) const
    -> ErrorOr<Nonnull<Expression*>> {
  std::list<ImplScope::ImplResult> maximals;
  for (const ImplResult& r : impls) {
    std::list<ImplScope::ImplResult> new_maximals;
    bool include_r = true;
    for (const ImplResult& m : maximals) {
      switch (
          MoreSpecificImpl(r, m, source_loc, original_scope, type_checker)) {
        case ImplComparison::MoreSpecific:
          // Don't add m to new_maximals.
          break;
        case ImplComparison::LessSpecific:
          new_maximals.push_back(m);
          include_r = false;
          break;
        case ImplComparison::EquallySpecific:
        case ImplComparison::Incomparable:
          new_maximals.push_back(m);
          break;
      }
    }
    if (include_r) {
      new_maximals.push_back(r);
    }
    maximals = std::move(new_maximals);
  }
  if (maximals.size() == 0) {
    return CompilationError(source_loc) << "could not find implementation of "
                                        << *iface_type << " for " << *impl_type;
  } else if (maximals.size() == 1) {
    return maximals.front().impl_expression;
  } else {
    return CompilationError(source_loc) << "ambiguous implementations of "
                                        << *iface_type << " for " << *impl_type;
  }
}

void ImplScope::TryResolve(Nonnull<const Value*> iface_type,
                           Nonnull<const Value*> type,
                           SourceLocation source_loc,
                           const ImplScope& original_scope,
                           const TypeChecker& type_checker,
                           std::list<ImplResult>& result) const {
  ResolveHere(iface_type, type, source_loc, original_scope, type_checker,
              result);
  for (Nonnull<const ImplScope*> parent : parent_scopes_) {
    parent->TryResolve(iface_type, type, source_loc, original_scope,
                       type_checker, result);
  }
}

void ImplScope::ResolveHere(Nonnull<const Value*> iface_type,
                            Nonnull<const Value*> impl_type,
                            SourceLocation source_loc,
                            const ImplScope& original_scope,
                            const TypeChecker& type_checker,
                            std::list<ImplResult>& result) const {
  if (iface_type->kind() != Value::Kind::InterfaceType) {
    CARBON_FATAL() << "expected an interface, not " << *iface_type;
  }
  const auto& iface = cast<InterfaceType>(*iface_type);
  for (const Impl& impl : impls_) {
    std::optional<Nonnull<Expression*>> m = type_checker.MatchImpl(
        iface, impl_type, impl, original_scope, source_loc);
    if (m.has_value()) {
      result.push_front({*m, impl});
    }
  }
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
