// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/syntax/parse.h"

#include <iostream>

#include "executable_semantics/syntax/driver.h"
#include "executable_semantics/syntax/parser.h"
#include "executable_semantics/tracing_flag.h"

extern FILE* yyin;

namespace Carbon {

// Returns an abstract representation of the program contained in the
// well-formed input file, or if the file was malformed, a description of the
// problem.
auto parse(const std::string& inputFileName)
    -> std::variant<AST, SyntaxErrorCode> {
  yyin = fopen(inputFileName.c_str(), "r");
  if (yyin == nullptr) {
    std::cerr << "Error opening '" << inputFileName
              << "': " << std::strerror(errno) << std::endl;
    exit(1);
  }

  std::optional<AST> parsedInput = std::nullopt;
  SyntaxDriver driver(inputFileName);
  auto syntaxErrorCode = yy::parser(parsedInput, driver)();
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

}  // namespace Carbon
