// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Helpers should be added here when logic in syntax.ypp is more than a single
// statement. The intent is to minimize the amount of C++ in the .ypp file, to
// improve ease of maintenance.

#ifndef CARBON_EXPLORER_INTERPRETER_EXEC_PROGRAM_H_
#define CARBON_EXPLORER_INTERPRETER_EXEC_PROGRAM_H_

#include "explorer/ast/ast.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

// Perform semantic analysis on the AST.
auto AnalyzeProgram(Nonnull<Arena*> arena, AST ast,
                    std::optional<Nonnull<llvm::raw_ostream*>> trace_stream)
    -> ErrorOr<AST>;

// Run the program's `Main` function.
auto ExecProgram(Nonnull<Arena*> arena, AST ast,
                 std::optional<Nonnull<llvm::raw_ostream*>> trace_stream)
    -> ErrorOr<int>;

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_EXEC_PROGRAM_H_
