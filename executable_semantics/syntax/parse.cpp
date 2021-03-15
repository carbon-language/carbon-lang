// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/syntax/parse.h"

#include <iostream>

#include "executable_semantics/syntax/parse_and_lex_context.h"
#include "executable_semantics/syntax/parser.h"
#include "executable_semantics/tracing_flag.h"

extern FILE* yyin;

namespace Carbon {

// Returns an abstract representation of the program contained in the
// well-formed input file, or if the file was malformed, a description of the
// problem.
auto Parse(const std::string& input_filename)
    -> std::variant<AST, SyntaxErrorCode> {
  yyin = fopen(input_filename.c_str(), "r");
  if (yyin == nullptr) {
    std::cerr << "Error opening '" << input_filename
              << "': " << std::strerror(errno) << std::endl;
    exit(1);
  }

  std::optional<AST> parsed_input = std::nullopt;
  ParseAndLexContext context(input_filename);

  auto syntax_error_code = yy::parser(parsed_input, context)();
  if (syntax_error_code != 0) {
    return syntax_error_code;
  }

  if (parsed_input == std::nullopt) {
    std::cerr << "Internal error: parser validated syntax yet didn't produce "
                 "an AST.\n";
    exit(1);
  }
  return *parsed_input;
}

}  // namespace Carbon
