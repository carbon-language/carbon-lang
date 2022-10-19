// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/static_scope.h"

#include <optional>

#include "explorer/common/error_builders.h"
#include "llvm/Support/Error.h"

namespace Carbon {

auto StaticScope::Add(const std::string& name, ValueNodeView entity,
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

void StaticScope::MarkDeclared(const std::string& name) {
  auto it = declared_names_.find(name);
  CARBON_CHECK(it != declared_names_.end()) << name << " not found";
  if (it->second.status == NameStatus::KnownButNotDeclared) {
    it->second.status = NameStatus::DeclaredButNotUsable;
  }
}

void StaticScope::MarkUsable(const std::string& name) {
  auto it = declared_names_.find(name);
  CARBON_CHECK(it != declared_names_.end()) << name << " not found";
  it->second.status = NameStatus::Usable;
}

auto StaticScope::Resolve(const std::string& name,
                          SourceLocation source_loc) const
    -> ErrorOr<ValueNodeView> {
  CARBON_ASSIGN_OR_RETURN(std::optional<ValueNodeView> result,
                          TryResolve(name, source_loc));
  if (!result) {
    return ProgramError(source_loc) << "could not resolve '" << name << "'";
  }
  return *result;
}

auto StaticScope::TryResolve(const std::string& name,
                             SourceLocation source_loc) const
    -> ErrorOr<std::optional<ValueNodeView>> {
  auto it = declared_names_.find(name);
  if (it != declared_names_.end()) {
    switch (it->second.status) {
      case NameStatus::KnownButNotDeclared:
        return ProgramError(source_loc)
               << "'" << name << "' has not been declared yet";
      case NameStatus::DeclaredButNotUsable:
        return ProgramError(source_loc)
               << "'" << name
               << "' is not usable until after it has been completely declared";
      case NameStatus::Usable:
        return std::make_optional(it->second.entity);
    }
  }
  std::optional<ValueNodeView> result;
  for (Nonnull<const StaticScope*> parent : parent_scopes_) {
    CARBON_ASSIGN_OR_RETURN(std::optional<ValueNodeView> parent_result,
                            parent->TryResolve(name, source_loc));
    if (parent_result.has_value() && result.has_value() &&
        *parent_result != *result) {
      return ProgramError(source_loc)
             << "'" << name << "' is ambiguous between "
             << result->base().source_loc() << " and "
             << parent_result->base().source_loc();
    }
    result = parent_result;
  }
  return result;
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
  if (returned_var_def_view_.has_value()) {
    return returned_var_def_view_;
  }
  for (Nonnull<const StaticScope*> parent : parent_scopes_) {
    std::optional<ValueNodeView> parent_returned_var =
        parent->ResolveReturned();
    if (parent_returned_var.has_value()) {
      return parent_returned_var;
    }
  }
  return std::nullopt;
}

}  // namespace Carbon
