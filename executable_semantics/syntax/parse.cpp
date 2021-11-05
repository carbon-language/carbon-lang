// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/syntax/parse.h"

#include "common/check.h"
#include "executable_semantics/common/error.h"
#include "executable_semantics/syntax/lexer.h"
#include "executable_semantics/syntax/parse_and_lex_context.h"
#include "executable_semantics/syntax/parser.h"

namespace Carbon {

auto Parse(Nonnull<Arena*> arena, const std::string& input_file_name,
           bool trace_carbon, bool trace_bison)
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
  std::optional<AST> ast = std::nullopt;
  ParseAndLexContext context(arena->New<std::string>(input_file_name),
                             trace_carbon);

  // Do the parse.
  auto parser = Parser(arena, scanner, context, &ast);
  if (trace_bison) {
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
  CHECK(ast != std::nullopt)
      << "parser validated syntax yet didn't produce an AST.";
  return *ast;
}

}  // namespace Carbon
