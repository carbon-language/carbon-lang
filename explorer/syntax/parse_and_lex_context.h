// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_SYNTAX_PARSE_AND_LEX_CONTEXT_H_
#define CARBON_EXPLORER_SYNTAX_PARSE_AND_LEX_CONTEXT_H_

#include <variant>

#include "explorer/ast/ast.h"
#include "explorer/syntax/parser.h"  // from parser.ypp

namespace Carbon {

// The state and functionality that is threaded "globally" through the
// lexing/parsing process.
class ParseAndLexContext {
 public:
  // Creates an instance analyzing the given input file.
  ParseAndLexContext(Nonnull<const std::string*> input_file_name,
                     bool parser_debug)
      : input_file_name_(input_file_name), parser_debug_(parser_debug) {}

  // Formats ands records a lexing oor parsing error. Returns an error token as
  // a convenience.
  auto RecordSyntaxError(Error error) -> Parser::symbol_type;
  auto RecordSyntaxError(const std::string& message) -> Parser::symbol_type;

  auto source_loc() const -> SourceLocation {
    return SourceLocation(input_file_name_,
                          static_cast<int>(current_token_position.begin.line));
  }

  auto parser_debug() const -> bool { return parser_debug_; }

  // The source range of the token being (or just) lex'd.
  location current_token_position;

  auto take_errors() -> std::vector<Error> {
    std::vector<Error> errors = std::move(errors_);
    errors_.clear();
    return errors;
  }

 private:
  // A path to the file processed, relative to the current working directory
  // when *this is called.
  Nonnull<const std::string*> input_file_name_;

  bool parser_debug_;

  std::vector<Error> errors_;
};

}  // namespace Carbon

// Gives flex the yylex prototype we want.
#define YY_DECL                                                         \
  auto yylex(Carbon::Nonnull<Carbon::Arena*> arena, yyscan_t yyscanner, \
             Carbon::ParseAndLexContext& context)                       \
      ->Carbon::Parser::symbol_type

// Declares yylex for the parser's sake.
YY_DECL;

#endif  // CARBON_EXPLORER_SYNTAX_PARSE_AND_LEX_CONTEXT_H_
