// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/static_scope.h"

#include "executable_semantics/common/error.h"

namespace Carbon {

void StaticScope::Add(std::string name, NamedEntityView entity) {
  auto [it, success] = declared_names_.insert({name, entity});
  if (!success && it->second != entity) {
    FATAL_COMPILATION_ERROR(entity.base().source_loc())
        << "Duplicate name `" << name << "` also found at "
        << it->second.base().source_loc();
  }
}

auto StaticScope::Resolve(const std::string& name,
                          SourceLocation source_loc) const -> NamedEntityView {
  std::optional<NamedEntityView> result = TryResolve(name, source_loc);
  if (!result.has_value()) {
    FATAL_COMPILATION_ERROR(source_loc) << "could not resolve '" << name << "'";
  }
  return *result;
}

auto StaticScope::TryResolve(const std::string& name,
                             SourceLocation source_loc) const
    -> std::optional<NamedEntityView> {
  auto it = declared_names_.find(name);
  if (it != declared_names_.end()) {
    return it->second;
  }
  std::optional<NamedEntityView> result;
  for (Nonnull<const StaticScope*> parent : parent_scopes_) {
    auto parent_result = parent->TryResolve(name, source_loc);
    if (parent_result.has_value() && result.has_value() &&
        *parent_result != *result) {
      FATAL_COMPILATION_ERROR(source_loc)
          << "'" << name << "' is ambiguous between "
          << result->base().source_loc() << " and "
          << parent_result->base().source_loc();
    }
    result = parent_result;
  }
  return result;
}

}  // namespace Carbon
