// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_INTERPRETER_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_INTERPRETER_H_

#include <optional>
#include <utility>
#include <vector>

#include "common/ostream.h"
#include "executable_semantics/ast/ast.h"
#include "executable_semantics/ast/declaration.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/pattern.h"
#include "executable_semantics/interpreter/action.h"
#include "executable_semantics/interpreter/action_stack.h"
#include "executable_semantics/interpreter/heap.h"
#include "executable_semantics/interpreter/value.h"
#include "llvm/ADT/ArrayRef.h"

namespace Carbon {

// Interprets the program defined by `ast`, allocating values on `arena` and
// printing traces if `trace` is true.
auto InterpProgram(const AST& ast, Nonnull<Arena*> arena, bool trace) -> int;

// Interprets `e` at compile-time, allocating values on `arena` and
// printing traces if `trace` is true. The caller must ensure that all the
// code this evaluates has been typechecked.
auto InterpExp(Nonnull<const Expression*> e, Nonnull<Arena*> arena, bool trace)
    -> Nonnull<const Value*>;

// Interprets `p` at compile-time, allocating values on `arena` and
// printing traces if `trace` is true. The caller must ensure that all the
// code this evaluates has been typechecked.
auto InterpPattern(Nonnull<const Pattern*> p, Nonnull<Arena*> arena, bool trace)
    -> Nonnull<const Value*>;

// Attempts to match `v` against the pattern `p`, returning whether matching
// is successful. If it is, populates **bindings with the variables bound by
// the match; `bindings` should only be nullopt in contexts where `p`
// is not permitted to bind variables. **bindings may be modified even if the
// match is unsuccessful, so it should typically be created for the
// PatternMatch call and then merged into an existing scope on success.
// TODO: consider moving this to a separate header.
[[nodiscard]] auto PatternMatch(Nonnull<const Value*> p,
                                Nonnull<const Value*> v,
                                SourceLocation source_loc,
                                std::optional<Nonnull<RuntimeScope*>> bindings)
    -> bool;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_INTERPRETER_H_
