// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>
#include <cstring>
#include <iostream>

#include "executable_semantics/syntax/driver.h"
#include "executable_semantics/syntax/syntax_helpers.h"
#include "executable_semantics/tracing_flag.h"
#include "llvm/Support/CommandLine.h"

int main(int argc, char* argv[]) {
  // yydebug = 1;

  using llvm::cl::desc;
  using llvm::cl::opt;
  opt<bool> quietOption("quiet", desc("Disable tracing"));
  opt<std::string> inputFileName(llvm::cl::Positional, desc("<input file>"),
                                 llvm::cl::Required);

  llvm::cl::ParseCommandLineOptions(argc, argv);
  if (quietOption) {
    Carbon::tracing_output = false;
  }

  auto analyzeSyntax = Carbon::SyntaxDriver(inputFileName);
  auto astOrError = analyzeSyntax();

  if (auto error = std::get_if<Carbon::SyntaxDriver::Error>(&astOrError)) {
    // Diagnostic already reported to std::cerr; this is just a return code.
    return *error;
  }

  // Typecheck and run the parsed program.
  Carbon::ExecProgram(std::get<Carbon::AST>(astOrError));
}
