// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_INTERPRETER_H_
#define CARBON_EXPLORER_INTERPRETER_INTERPRETER_H_

#include "common/ostream.h"
#include "explorer/ast/ast.h"
#include "explorer/ast/expression.h"
#include "explorer/ast/value.h"
#include "explorer/base/trace_stream.h"

namespace Carbon {

// Interprets the program defined by `ast`, allocating values on `arena` and
// printing traces if `trace` is true.
auto InterpProgram(const AST& ast, Nonnull<Arena*> arena,
                   Nonnull<TraceStream*> trace_stream,
                   Nonnull<llvm::raw_ostream*> print_stream) -> ErrorOr<int>;

// Interprets `e` at compile-time, allocating values on `arena` and
// printing traces if `trace` is true. The caller must ensure that all the
// code this evaluates has been typechecked.
auto InterpExp(Nonnull<const Expression*> e, Nonnull<Arena*> arena,
               Nonnull<TraceStream*> trace_stream,
               Nonnull<llvm::raw_ostream*> print_stream)
    -> ErrorOr<Nonnull<const Value*>>;
}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_INTERPRETER_H_
