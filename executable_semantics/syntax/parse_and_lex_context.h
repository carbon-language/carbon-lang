// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_SYNTAX_DRIVER_H_
#define EXECUTABLE_SEMANTICS_SYNTAX_DRIVER_H_

#include <variant>

#include "executable_semantics/ast/ast.h"
#include "executable_semantics/syntax/parser.h"  // from parser.ypp

namespace Carbon {

// The state and functionality that is threaded "globally" through the
// lexing/parsing process.
class ParseAndLexContext {
 public:
  // Creates an instance analyzing the given input file.
  ParseAndLexContext(const std::string& input_file)
      : input_file_name(global_arena->New<std::string>(input_file)) {}

  // Writes a syntax error diagnostic containing message to standard error.
  auto PrintDiagnostic(const std::string& message) -> void;

  auto SourceLoc() -> SourceLocation {
    return SourceLocation(input_file_name,
                          static_cast<int>(current_token_position.begin.line));
  }

  // The source range of the token being (or just) lex'd.
  location current_token_position;

 private:
  // A path to the file processed, relative to the current working directory
  // when *this is called.
  Ptr<const std::string> input_file_name;
};

}  // namespace Carbon

// Gives flex the yylex prototype we want.
#define YY_DECL \
  Carbon::Parser::symbol_type yylex(Carbon::ParseAndLexContext& context)

// Declares yylex for the parser's sake.
YY_DECL;

#endif  // EXECUTABLE_SYNTAX_DRIVER_H_
