// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/name_scope.h"

namespace Carbon::Check {

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
  return {{.class_decl = *class_decl, .name_scope = &scope}};
}

auto TryAsInterfaceScope(Context& context, SemIR::NameScopeId scope_id)
    -> std::optional<InterfaceScope> {
  if (!scope_id.has_value()) {
    return std::nullopt;
  }
  auto& scope = context.name_scopes().Get(scope_id);
  if (!scope.inst_id().has_value()) {
    return std::nullopt;
  }
  auto interface_decl =
      context.insts().TryGetAs<SemIR::InterfaceDecl>(scope.inst_id());
  if (!interface_decl) {
    return std::nullopt;
  }
  return {{.interface_decl = *interface_decl, .name_scope = &scope}};
}

auto TryAsNamedConstraintScope(Context& context, SemIR::NameScopeId scope_id)
    -> std::optional<NamedConstraintScope> {
  if (!scope_id.has_value()) {
    return std::nullopt;
  }
  auto& scope = context.name_scopes().Get(scope_id);
  if (!scope.inst_id().has_value()) {
    return std::nullopt;
  }
  auto constraint_decl =
      context.insts().TryGetAs<SemIR::NamedConstraintDecl>(scope.inst_id());
  if (!constraint_decl) {
    return std::nullopt;
  }
  return {{.constraint_decl = *constraint_decl, .name_scope = &scope}};
}

}  // namespace Carbon::Check
