// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPLORER_INTERPRETER_RESOLVE_CONTROL_FLOW_H_
#define EXPLORER_INTERPRETER_RESOLVE_CONTROL_FLOW_H_

#include "explorer/ast/ast.h"
#include "explorer/common/nonnull.h"

namespace Carbon {

// Resolves non-local control-flow edges, such as `break` and `return`, in the
// given AST.
// On failure, `ast` is left in a partial state and should not be further
// processed.
auto ResolveControlFlow(AST& ast) -> ErrorOr<Success>;

}  // namespace Carbon

#endif  // EXPLORER_INTERPRETER_RESOLVE_CONTROL_FLOW_H_
