// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_RESOLVE_UNFORMED_H_
#define CARBON_EXPLORER_INTERPRETER_RESOLVE_UNFORMED_H_

#include "explorer/ast/ast.h"
#include "explorer/common/nonnull.h"

namespace Carbon {

// An intraprocedural forward analysis that checks the may-be-formed states on
// local variables. Returns compilation error on usage of must-be-unformed
// variables.
auto ResolveUnformed(const AST& ast) -> ErrorOr<Success>;

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_RESOLVE_UNFORMED_H_
