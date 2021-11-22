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

auto ParseImpl(yyscan_t scanner, Nonnull<Arena*> arena,
               const std::string& input_file_name, bool trace)
    -> std::variant<AST, SyntaxErrorCode> {
  // Prepare other parser arguments.
  std::optional<AST> ast = std::nullopt;
  ParseAndLexContext context(arena->New<std::string>(input_file_name), trace);

  // Do the parse.
  auto parser = Parser(arena, scanner, context, &ast);
  if (trace) {
    parser.set_debug_level(1);
  }
  auto syntax_error_code = parser();

  // Return an error if appropriate.
  if (syntax_error_code != 0) {
    return syntax_error_code;
  }

  // Return parse results.
  CHECK(ast != std::nullopt)
      << "parser validated syntax yet didn't produce an AST.";
  return *ast;
}

auto Parse(Nonnull<Arena*> arena, const std::string& input_file_name,
           bool trace) -> std::variant<AST, SyntaxErrorCode> {
  FILE* input_file = fopen(input_file_name.c_str(), "r");
  if (input_file == nullptr) {
    FATAL_PROGRAM_ERROR_NO_LINE() << "Error opening '" << input_file_name
                                  << "': " << std::strerror(errno);
  }

  // Prepare the lexer.
  yyscan_t scanner;
  yylex_init(&scanner);
  auto buffer = yy_create_buffer(input_file, YY_BUF_SIZE, scanner);
  yy_switch_to_buffer(buffer, scanner);

  std::variant<AST, SyntaxErrorCode> result =
      ParseImpl(scanner, arena, input_file_name, trace);

  // Clean up the lexer.
  fclose(input_file);
  yy_delete_buffer(buffer, scanner);
  yylex_destroy(scanner);

  return result;
}

auto ParseFromString(Nonnull<Arena*> arena, const std::string& input_file_name,
                     std::string_view file_contents, bool trace)
    -> std::variant<Carbon::AST, SyntaxErrorCode> {
  // Prepare the lexer.
  yyscan_t scanner;
  yylex_init(&scanner);
  auto buffer =
      yy_scan_bytes(file_contents.data(), file_contents.size(), scanner);
  yy_switch_to_buffer(buffer, scanner);

  std::variant<AST, SyntaxErrorCode> result =
      ParseImpl(scanner, arena, input_file_name, trace);

  // Clean up the lexer.
  yy_delete_buffer(buffer, scanner);
  yylex_destroy(scanner);

  return result;
}
}  // namespace Carbon
