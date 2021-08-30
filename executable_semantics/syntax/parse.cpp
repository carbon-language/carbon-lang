// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/syntax/parse.h"

#include "common/check.h"
#include "executable_semantics/common/error.h"
#include "executable_semantics/common/tracing_flag.h"
#include "executable_semantics/syntax/lexer.h"
#include "executable_semantics/syntax/parse_and_lex_context.h"
#include "executable_semantics/syntax/parser.h"

namespace Carbon {

// Returns an abstract representation of the program contained in the
// well-formed input file, or if the file was malformed, a description of the
// problem.
auto Parse(const std::string& input_file_name)
    -> std::variant<AST, SyntaxErrorCode> {
  FILE* input_file = fopen(input_file_name.c_str(), "r");
  if (input_file == nullptr) {
    FATAL_PROGRAM_ERROR_NO_LINE() << "Error opening '" << input_file_name
                                  << "': " << std::strerror(errno);
  }

  // Prepare the lexer.
  yyscan_t scanner;
  yylex_init(&scanner);
  yyset_in(input_file, scanner);

  // Prepare other parser arguments.
  std::optional<AST> parsed_input = std::nullopt;
  ParseAndLexContext context(input_file_name);

  // Do the parse.
  auto parser = Parser(parsed_input, scanner, context);
  if (tracing_output) {
    parser.set_debug_level(1);
  }
  auto syntax_error_code = parser();

  // Clean up the lexer.
  fclose(input_file);
  yylex_destroy(scanner);

  // Return an error if appropriate.
  if (syntax_error_code != 0) {
    return syntax_error_code;
  }

  // Return parse results.
  CHECK(parsed_input != std::nullopt)
      << "parser validated syntax yet didn't produce an AST.";
  return *parsed_input;
}

}  // namespace Carbon
