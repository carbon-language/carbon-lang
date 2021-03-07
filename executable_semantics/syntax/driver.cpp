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
  auto syntaxErrorCode = yyparse(parsedInput, *this);
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
