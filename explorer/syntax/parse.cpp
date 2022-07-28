// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/syntax/parse.h"

#include "common/check.h"
#include "common/error.h"
#include "explorer/common/error_builders.h"
#include "explorer/syntax/lexer.h"
#include "explorer/syntax/parse_and_lex_context.h"
#include "explorer/syntax/parser.h"
#include "llvm/Support/Error.h"

namespace Carbon {

static auto ParseImpl(yyscan_t scanner, Nonnull<Arena*> arena,
                      std::string_view input_file_name, bool parser_debug)
    -> ErrorOr<AST> {
  // Prepare other parser arguments.
  std::optional<AST> ast = std::nullopt;
  ParseAndLexContext context(arena->New<std::string>(input_file_name),
                             parser_debug);

  // Do the parse.
  auto parser = Parser(arena, scanner, context, &ast);
  if (parser_debug) {
    parser.set_debug_level(1);
  }

  if (auto syntax_error_code = parser(); syntax_error_code != 0) {
    auto errors = context.take_errors();
    if (errors.empty()) {
      return Error("Unknown parser erroor");
    }
    return std::move(errors.front());
  }

  // Return parse results.
  CARBON_CHECK(ast != std::nullopt)
      << "parser validated syntax yet didn't produce an AST.";
  return *ast;
}

auto Parse(Nonnull<Arena*> arena, std::string_view input_file_name,
           bool parser_debug) -> ErrorOr<AST> {
  std::string name_str(input_file_name);
  FILE* input_file = fopen(name_str.c_str(), "r");
  if (input_file == nullptr) {
    return ProgramError(SourceLocation(name_str.c_str(), 0))
           << "Error opening file: " << std::strerror(errno);
  }

  // Prepare the lexer.
  yyscan_t scanner;
  yylex_init(&scanner);
  auto buffer = yy_create_buffer(input_file, YY_BUF_SIZE, scanner);
  yy_switch_to_buffer(buffer, scanner);

  ErrorOr<AST> result =
      ParseImpl(scanner, arena, input_file_name, parser_debug);

  // Clean up the lexer.
  yy_delete_buffer(buffer, scanner);
  yylex_destroy(scanner);
  fclose(input_file);

  return result;
}

auto ParseFromString(Nonnull<Arena*> arena, std::string_view input_file_name,
                     std::string_view file_contents, bool parser_debug)
    -> ErrorOr<AST> {
  // Prepare the lexer.
  yyscan_t scanner;
  yylex_init(&scanner);
  auto buffer =
      yy_scan_bytes(file_contents.data(), file_contents.size(), scanner);
  yy_switch_to_buffer(buffer, scanner);

  ErrorOr<AST> result =
      ParseImpl(scanner, arena, input_file_name, parser_debug);

  // Clean up the lexer.
  yy_delete_buffer(buffer, scanner);
  yylex_destroy(scanner);

  return result;
}
}  // namespace Carbon
