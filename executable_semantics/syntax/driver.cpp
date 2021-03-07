// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/syntax/driver.h"

#include <cstring>
#include <iostream>

#include "executable_semantics/tracing_flag.h"

extern FILE* yyin;

// Returns an abstract representation of the program contained in the
// well-formed input file, or if the file was malformed, a description of the
// problem.
auto Carbon::SyntaxDriver::operator()()
    -> std::variant<Carbon::AST, Carbon::SyntaxDriver::Error> {
  yyin = fopen(inputFileName.c_str(), "r");
  if (yyin == nullptr) {
    std::cerr << "Error opening '" << inputFileName
              << "': " << std::strerror(errno) << std::endl;
    exit(1);
  }

  std::optional<Carbon::AST> parsedInput = std::nullopt;
  yy::parser parse(parsedInput, *this);
  auto syntaxErrorCode = parse();
  if (syntaxErrorCode != 0) {
    return syntaxErrorCode;
  }

  if (parsedInput == std::nullopt) {
    std::cerr << "Internal error: parser validated syntax yet didn't produce "
                 "an AST.\n";
    exit(1);
  }
  return *parsedInput;
}

// Writes a syntax error diagnostic, containing message, for the input file at
// the given line, to standard error.
auto Carbon::SyntaxDriver::PrintDiagnostic(const char* message, int line_num)
    -> void {
  std::cerr << inputFileName << ":" << line_num << ": " << message << std::endl;
  exit(-1);  // TODO: do we really want this here?  It makes the comment and the
             // name a lie, and renders some of the other yyparse() result
             // propagation code moot.
}
