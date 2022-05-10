// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPLORER_SYNTAX_DRIVER_H_
#define EXPLORER_SYNTAX_DRIVER_H_

#include <variant>

#include "explorer/ast/ast.h"
#include "explorer/syntax/parser.h"  // from parser.ypp

namespace Carbon {

// The state and functionality that is threaded "globally" through the
// lexing/parsing process.
class ParseAndLexContext {
 public:
  // Creates an instance analyzing the given input file.
  ParseAndLexContext(Nonnull<const std::string*> input_file_name, bool trace)
      : input_file_name_(input_file_name), trace_(trace) {}

  // Formats ands records a lexer error. Returns an error token as a
  // convenience.
  auto RecordSyntaxError(const std::string& message,
                         bool prefix_with_newline = false)
      -> Parser::symbol_type;

  auto source_loc() const -> SourceLocation {
    return SourceLocation(input_file_name_,
                          static_cast<int>(current_token_position.begin.line));
  }

  auto trace() const -> bool { return trace_; }

  // The source range of the token being (or just) lex'd.
  location current_token_position;

  auto error_messages() const -> const std::vector<std::string> {
    return error_messages_;
  }

 private:
  // A path to the file processed, relative to the current working directory
  // when *this is called.
  Nonnull<const std::string*> input_file_name_;

  bool trace_;

  std::vector<std::string> error_messages_;
};

}  // namespace Carbon

// Gives flex the yylex prototype we want.
#define YY_DECL                                                         \
  auto yylex(Carbon::Nonnull<Carbon::Arena*> arena, yyscan_t yyscanner, \
             Carbon::ParseAndLexContext& context)                       \
      ->Carbon::Parser::symbol_type

// Declares yylex for the parser's sake.
YY_DECL;

#endif  // EXECUTABLE_SYNTAX_DRIVER_H_
