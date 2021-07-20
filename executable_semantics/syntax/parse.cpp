// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/syntax/parse.h"

#include <iostream>

#include "executable_semantics/common/tracing_flag.h"
#include "executable_semantics/syntax/parse_and_lex_context.h"
#include "executable_semantics/syntax/parser.h"

extern FILE* yyin;

namespace Carbon {

// Returns an abstract representation of the program contained in the
// well-formed input file, or if the file was malformed, a description of the
// problem.
auto parse(const std::string& input_file_name)
    -> std::variant<AST, SyntaxErrorCode> {
  yyin = fopen(input_file_name.c_str(), "r");
  if (yyin == nullptr) {
    std::cerr << "Error opening '" << input_file_name
              << "': " << std::strerror(errno) << std::endl;
    exit(1);
  }

  std::optional<AST> parsed_input = std::nullopt;
  ParseAndLexContext context(input_file_name);

  auto parser = yy::parser(parsed_input, context);
  if (tracing_output) {
    parser.set_debug_level(1);
  }
  auto syntax_error_code = parser();
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
