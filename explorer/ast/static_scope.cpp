// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/static_scope.h"

#include <optional>

#include "explorer/common/error_builders.h"
#include "llvm/Support/Error.h"

namespace Carbon {

auto StaticScope::Add(std::string_view name, ValueNodeView entity,
                      NameStatus status) -> ErrorOr<Success> {
  auto [it, inserted] = declared_names_.insert({name, {entity, status}});
  if (!inserted) {
    if (it->second.entity != entity) {
      return ProgramError(entity.base().source_loc())
             << "Duplicate name `" << name << "` also found at "
             << it->second.entity.base().source_loc();
    }
    if (static_cast<int>(status) > static_cast<int>(it->second.status)) {
      it->second.status = status;
    }
  }
  return Success();
}

void StaticScope::MarkDeclared(std::string_view name) {
  auto it = declared_names_.find(name);
  CARBON_CHECK(it != declared_names_.end()) << name << " not found";
  if (it->second.status == NameStatus::KnownButNotDeclared) {
    it->second.status = NameStatus::DeclaredButNotUsable;
  }
}

void StaticScope::MarkUsable(std::string_view name) {
  auto it = declared_names_.find(name);
  CARBON_CHECK(it != declared_names_.end()) << name << " not found";
  it->second.status = NameStatus::Usable;
}

auto StaticScope::Resolve(std::string_view name,
                          SourceLocation source_loc) const
    -> ErrorOr<ValueNodeView> {
  CARBON_ASSIGN_OR_RETURN(std::optional<ValueNodeView> result,
                          TryResolve(name, source_loc));
  if (!result) {
    return ProgramError(source_loc) << "could not resolve '" << name << "'";
  }
  return *result;
}

auto StaticScope::ResolveHere(std::optional<ValueNodeView> this_scope,
                              std::string_view name, SourceLocation source_loc,
                              bool allow_undeclared) const
    -> ErrorOr<ValueNodeView> {
  CARBON_ASSIGN_OR_RETURN(std::optional<ValueNodeView> result,
                          TryResolveHere(name, source_loc, allow_undeclared));
  if (!result) {
    if (this_scope) {
      return ProgramError(source_loc)
             << "name '" << name << "' has not been declared in "
             << PrintAsID(this_scope->base());
    } else {
      return ProgramError(source_loc)
             << "name '" << name << "' has not been declared in this scope";
    }
  }
  return *result;
}

auto StaticScope::TryResolve(std::string_view name,
                             SourceLocation source_loc) const
    -> ErrorOr<std::optional<ValueNodeView>> {
  for (const StaticScope* scope = this; scope;
       scope = scope->parent_scope_.value_or(nullptr)) {
    CARBON_ASSIGN_OR_RETURN(
        std::optional<ValueNodeView> value,
        scope->TryResolveHere(name, source_loc, /*allow_undeclared=*/false));
    if (value) {
      return value;
    }
  }
  return {std::nullopt};
}

auto StaticScope::TryResolveHere(std::string_view name,
                                 SourceLocation source_loc,
                                 bool allow_undeclared) const
    -> ErrorOr<std::optional<ValueNodeView>> {
  auto it = declared_names_.find(name);
  if (it == declared_names_.end()) {
    return {std::nullopt};
  }
  if (allow_undeclared) {
    return {it->second.entity};
  }
  switch (it->second.status) {
    case NameStatus::KnownButNotDeclared:
      return ProgramError(source_loc)
             << "'" << name << "' has not been declared yet";
    case NameStatus::DeclaredButNotUsable:
      return ProgramError(source_loc) << "'" << name
                                      << "' is not usable until after it "
                                         "has been completely declared";
    case NameStatus::Usable:
      return {it->second.entity};
  }
}

auto StaticScope::AddReturnedVar(ValueNodeView returned_var_def_view)
    -> ErrorOr<Success> {
  std::optional<ValueNodeView> resolved_returned_var = ResolveReturned();
  if (resolved_returned_var.has_value()) {
    return ProgramError(returned_var_def_view.base().source_loc())
           << "Duplicate definition of returned var also found at "
           << resolved_returned_var->base().source_loc();
  }
  returned_var_def_view_ = std::move(returned_var_def_view);
  return Success();
}

auto StaticScope::ResolveReturned() const -> std::optional<ValueNodeView> {
  for (const StaticScope* scope = this; scope;
       scope = scope->parent_scope_.value_or(nullptr)) {
    if (scope->returned_var_def_view_.has_value()) {
      return scope->returned_var_def_view_;
    }
  }
  return std::nullopt;
}

}  // namespace Carbon
