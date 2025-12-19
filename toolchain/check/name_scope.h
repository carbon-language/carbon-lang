// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_NAME_SCOPE_H_
#define CARBON_TOOLCHAIN_CHECK_NAME_SCOPE_H_

#include <optional>

#include "common/concepts.h"
#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

struct ClassScope {
  SemIR::ClassDecl class_decl;
  SemIR::NameScope* name_scope;
};

// If the specified name scope corresponds to a class, returns the corresponding
// class declaration.
auto TryAsClassScope(Context& context, SemIR::NameScopeId scope_id)
    -> std::optional<ClassScope>;

struct InterfaceScope {
  SemIR::InterfaceDecl interface_decl;
  SemIR::NameScope* name_scope;
};

// If the specified name scope corresponds to a interface, returns the
// corresponding interface declaration.
auto TryAsInterfaceScope(Context& context, SemIR::NameScopeId scope_id)
    -> std::optional<InterfaceScope>;

struct NamedConstraintScope {
  SemIR::NamedConstraintDecl constraint_decl;
  SemIR::NameScope* name_scope;
};

// If the specified name scope corresponds to a named constraint, returns the
// corresponding constraint declaration.
auto TryAsNamedConstraintScope(Context& context, SemIR::NameScopeId scope_id)
    -> std::optional<NamedConstraintScope>;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_NAME_SCOPE_H_
