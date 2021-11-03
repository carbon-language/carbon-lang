// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_RESOLVE_NAMES_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_RESOLVE_NAMES_H_

#include "executable_semantics/ast/ast.h"
#include "executable_semantics/common/arena.h"

namespace Carbon {

// Resolves names (IdentifierExpressions) in the AST.
void ResolveNames(Nonnull<Arena*>, AST& ast);

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_RESOLVE_CONTROL_FLOW_H_
