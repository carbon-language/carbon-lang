// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/static_scope.h"

#include "executable_semantics/common/error.h"

namespace Carbon {

NamedEntity::~NamedEntity() = default;

void StaticScope::Add(std::string name, Nonnull<const NamedEntity*> entity) {
  if (!declared_names_.insert({name, entity}).second) {
    FATAL_COMPILATION_ERROR(entity->source_loc())
        << "Duplicate name `" << name << "` also found at "
        << declared_names_[name]->source_loc();
  }
}

auto StaticScope::Resolve(const std::string& name,
                          SourceLocation source_loc) const
    -> Nonnull<const NamedEntity*> {
  std::optional<Nonnull<const NamedEntity*>> result =
      TryResolve(name, source_loc);
  if (!result.has_value()) {
    FATAL_COMPILATION_ERROR(source_loc)
        << "'" << name << "' is not declared in this scope";
  }
  return *result;
}

auto StaticScope::TryResolve(const std::string& name,
                             SourceLocation source_loc) const
    -> std::optional<Nonnull<const NamedEntity*>> {
  auto it = declared_names_.find(name);
  if (it != declared_names_.end()) {
    return it->second;
  }
  std::optional<Nonnull<const NamedEntity*>> result;
  for (Nonnull<const StaticScope*> parent : parent_scopes_) {
    auto parent_result = parent->TryResolve(name, source_loc);
    if (parent_result.has_value() && result.has_value() &&
        *parent_result != *result) {
      FATAL_COMPILATION_ERROR(source_loc)
          << "'" << name << "' is ambiguous between " << (*result)->source_loc()
          << " and " << (*parent_result)->source_loc();
    }
    result = parent_result;
  }
  return result;
}

}  // namespace Carbon
