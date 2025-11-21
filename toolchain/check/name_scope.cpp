// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/name_scope.h"

namespace Carbon::Check {

// If the specified name scope corresponds to a class, returns the corresponding
// class declaration.
auto TryAsClassScope(Context& context, SemIR::NameScopeId scope_id)
    -> std::optional<ClassScope> {
  if (!scope_id.has_value()) {
    return std::nullopt;
  }
  auto& scope = context.name_scopes().Get(scope_id);
  if (!scope.inst_id().has_value()) {
    return std::nullopt;
  }
  auto class_decl = context.insts().TryGetAs<SemIR::ClassDecl>(scope.inst_id());
  if (!class_decl) {
    return std::nullopt;
  }
  // TODO: This is also valid in a mixin.
  return {{.class_decl = *class_decl}};
}

}  // namespace Carbon::Check
