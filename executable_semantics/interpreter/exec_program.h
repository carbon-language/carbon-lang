// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Helpers should be added here when logic in syntax.ypp is more than a single
// statement. The intent is to minimize the amount of C++ in the .ypp file, to
// improve ease of maintenance.

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_EXEC_PROGRAM_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_EXEC_PROGRAM_H_

#include "executable_semantics/ast/ast.h"

namespace Carbon {

// Runs the top-level declaration list.
void ExecProgram(Nonnull<Arena*> arena, AST ast, bool trace);

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_EXEC_PROGRAM_H_
