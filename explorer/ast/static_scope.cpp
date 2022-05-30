// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/static_scope.h"

#include "explorer/common/error_builders.h"
#include "llvm/Support/Error.h"

namespace Carbon {

auto StaticScope::Add(const std::string& name, ValueNodeView entity,
                      bool usable) -> ErrorOr<Success> {
  auto [it, inserted] = declared_names_.insert({name, {entity, usable}});
  if (!inserted) {
    if (it->second.entity != entity) {
      return CompilationError(entity.base().source_loc())
             << "Duplicate name `" << name << "` also found at "
             << it->second.entity.base().source_loc();
    }
    CARBON_CHECK(usable || !it->second.usable)
        << entity.base().source_loc() << " attempting to mark a usable name `"
        << name << "` as unusable";
  }
  return Success();
}

void StaticScope::MarkUsable(const std::string& name) {
  auto it = declared_names_.find(name);
  CARBON_CHECK(it != declared_names_.end()) << name << " not found";
  it->second.usable = true;
}

auto StaticScope::Resolve(const std::string& name,
                          SourceLocation source_loc) const
    -> ErrorOr<ValueNodeView> {
  CARBON_ASSIGN_OR_RETURN(std::optional<ValueNodeView> result,
                          TryResolve(name, source_loc));
  if (!result) {
    return CompilationError(source_loc) << "could not resolve '" << name << "'";
  }
  return *result;
}

auto StaticScope::TryResolve(const std::string& name,
                             SourceLocation source_loc) const
    -> ErrorOr<std::optional<ValueNodeView>> {
  auto it = declared_names_.find(name);
  if (it != declared_names_.end()) {
    if (!it->second.usable) {
      return CompilationError(source_loc)
             << "'" << name << "' is not usable in the current context";
    }
    return std::make_optional(it->second.entity);
  }
  std::optional<ValueNodeView> result;
  for (Nonnull<const StaticScope*> parent : parent_scopes_) {
    CARBON_ASSIGN_OR_RETURN(std::optional<ValueNodeView> parent_result,
                            parent->TryResolve(name, source_loc));
    if (parent_result.has_value() && result.has_value() &&
        *parent_result != *result) {
      return CompilationError(source_loc)
             << "'" << name << "' is ambiguous between "
             << result->base().source_loc() << " and "
             << parent_result->base().source_loc();
    }
    result = parent_result;
  }
  return result;
}

}  // namespace Carbon
