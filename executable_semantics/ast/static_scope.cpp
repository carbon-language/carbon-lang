// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/static_scope.h"

#include "executable_semantics/common/error.h"

namespace Carbon {

auto GetSourceLoc(NamedEntity entity) -> SourceLocation {
  return std::visit([](auto&& arg) { return arg->source_loc(); }, entity);
}

void StaticScope::Add(std::string name, NamedEntity entity) {
  if (!declared_names_.insert({name, entity}).second) {
    FATAL_COMPILATION_ERROR(GetSourceLoc(entity))
        << "Duplicate name `" << name << "` also found at "
        << GetSourceLoc(declared_names_[name]);
  }
}

}  // namespace Carbon
