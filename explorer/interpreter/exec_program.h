// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Helpers should be added here when logic in syntax.ypp is more than a single
// statement. The intent is to minimize the amount of C++ in the .ypp file, to
// improve ease of maintenance.

#ifndef EXPLORER_INTERPRETER_EXEC_PROGRAM_H_
#define EXPLORER_INTERPRETER_EXEC_PROGRAM_H_

#include "explorer/ast/ast.h"

namespace Carbon {

// Runs the top-level declaration list.
auto ExecProgram(Nonnull<Arena*> arena, AST ast, bool trace) -> ErrorOr<int>;

}  // namespace Carbon

#endif  // EXPLORER_INTERPRETER_EXEC_PROGRAM_H_
