// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_RESOLVE_CONTROL_FLOW_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_RESOLVE_CONTROL_FLOW_H_

#include "executable_semantics/ast/ast.h"
#include "executable_semantics/common/nonnull.h"

namespace Carbon {

// Resolves non-local control-flow edges, such as `break` and `return`, in the
// given AST.
void ResolveControlFlow(AST& ast);

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_RESOLVE_CONTROL_FLOW_H_
