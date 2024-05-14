// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/syntax/parse.h"

#include "common/check.h"
#include "common/error.h"
#include "explorer/base/error_builders.h"
#include "explorer/syntax/lexer.h"
#include "explorer/syntax/parse_and_lex_context.h"
#include "explorer/syntax/parser.h"

namespace Carbon {

static auto ParseImpl(yyscan_t scanner, Nonnull<Arena*> arena,
                      std::string_view input_file_name, FileKind file_kind,
                      bool parser_debug) -> ErrorOr<AST> {
  // Prepare other parser arguments.
  std::optional<AST> ast = std::nullopt;
  ParseAndLexContext context(arena->New<std::string>(input_file_name),
                             file_kind, parser_debug);

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

auto Parse(llvm::vfs::FileSystem& fs, Nonnull<Arena*> arena,
           std::string_view input_file_name, FileKind file_kind,
           bool parser_debug) -> ErrorOr<AST> {
  llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>> input_file =
      fs.openFileForRead(input_file_name);
  if (input_file.getError()) {
    return ProgramError(SourceLocation(input_file_name, 0, file_kind))
           << "Error opening file: " << input_file.getError().message();
  }

  llvm::ErrorOr<llvm::vfs::Status> status = (*input_file)->status();
  if (status.getError()) {
    return Error(status.getError().message());
  }
  auto size = status->getSize();
  if (size >= std::numeric_limits<int32_t>::max()) {
    return ProgramError(SourceLocation(input_file_name, 0, file_kind))
           << "File is over the 2GiB input limit.";
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      (*input_file)
          ->getBuffer(input_file_name, size, /*RequiresNullTerminator=*/false);
  if (buffer.getError()) {
    return Error(buffer.getError().message());
  }

  return ParseFromString(arena, input_file_name, file_kind,
                         (*buffer)->getBuffer(), parser_debug);
}

auto ParseFromString(Nonnull<Arena*> arena, std::string_view input_file_name,
                     FileKind file_kind, std::string_view file_contents,
                     bool parser_debug) -> ErrorOr<AST> {
  // Prepare the lexer.
  yyscan_t scanner;
  yylex_init(&scanner);
  auto* buffer =
      yy_scan_bytes(file_contents.data(), file_contents.size(), scanner);
  yy_switch_to_buffer(buffer, scanner);

  ErrorOr<AST> result =
      ParseImpl(scanner, arena, input_file_name, file_kind, parser_debug);

  // Clean up the lexer.
  yy_delete_buffer(buffer, scanner);
  yylex_destroy(scanner);

  return result;
}
}  // namespace Carbon
