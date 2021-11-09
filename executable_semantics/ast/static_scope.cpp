// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/static_scope.h"

#include "executable_semantics/common/error.h"

namespace Carbon {

NamedEntityInterface::~NamedEntityInterface() = default;

void StaticScope::Add(std::string name,
                      Nonnull<const NamedEntityInterface*> entity) {
  if (!declared_names_.insert({name, entity}).second) {
    FATAL_COMPILATION_ERROR(entity->source_loc())
        << "Duplicate name `" << name << "` also found at "
        << declared_names_[name]->source_loc();
  }
}

}  // namespace Carbon
