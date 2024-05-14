// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_RESOLVE_NAMES_H_
#define CARBON_EXPLORER_INTERPRETER_RESOLVE_NAMES_H_

#include "explorer/ast/ast.h"
#include "explorer/base/arena.h"
#include "explorer/base/trace_stream.h"

namespace Carbon {

// Resolves names (IdentifierExpressions) in the AST.
// On failure, `ast` is left in a partial state and should not be further
// processed.
auto ResolveNames(AST& ast, Nonnull<TraceStream*> trace_stream)
    -> ErrorOr<Success>;

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_RESOLVE_NAMES_H_
