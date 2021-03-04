// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Helpers should be added here when logic in syntax.ypp is more than a single
// statement. The intent is to minimize the amount of C++ in the .ypp file, to
// improve ease of maintenance.

#ifndef EXECUTABLE_SEMANTICS_EXEC_H_
#define EXECUTABLE_SEMANTICS_EXEC_H_

#include <list>

#include "executable_semantics/ast/declaration.h"
#include "executable_semantics/location.hh"

namespace Carbon {

// Initialized by main(), and used when printing errors.
extern const char* input_filename;

/// Reports a syntax error at `sourceLocation` with the given `details` and
/// exits the program with code `-1`.
void SyntaxError(std::string message, yy::location sourceLocation);

// Runs the top-level declaration list.
void ExecProgram(std::list<Declaration>* fs);

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_EXEC_H_
