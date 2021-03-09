// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/syntax/driver.h"

#include <cstring>
#include <iostream>

#include "executable_semantics/tracing_flag.h"

// Writes a syntax error diagnostic, containing message, for the input file at
// the given line, to standard error.
auto Carbon::SyntaxDriver::PrintDiagnostic(const char* message, int line_num)
    -> void {
  std::cerr << inputFileName << ":" << line_num << ": " << message << std::endl;
  exit(-1);  // TODO: do we really want this here?  It makes the comment and the
             // name a lie, and renders some of the other yyparse() result
             // propagation code moot.
}
