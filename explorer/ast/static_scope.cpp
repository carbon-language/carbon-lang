// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/static_scope.h"

#include "explorer/common/error_builders.h"
#include "llvm/Support/Error.h"

namespace Carbon {

auto StaticScope::Add(std::string name, ValueNodeView entity)
    -> ErrorOr<Success> {
  auto [it, success] = declared_names_.insert({name, entity});
  if (!success && it->second != entity) {
    return CompilationError(entity.base().source_loc())
           << "Duplicate name `" << name << "` also found at "
           << it->second.base().source_loc();
  }
  return Success();
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
    return std::make_optional(it->second);
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
