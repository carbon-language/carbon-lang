// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_SYNTAX_DRIVER_H_
#define EXECUTABLE_SEMANTICS_SYNTAX_DRIVER_H_

#include <variant>

#include "executable_semantics/ast/abstract_syntax_tree.h"
#include "executable_semantics/syntax/parser.h"  // from parser.ypp

namespace Carbon {

// The state and functionality that is threaded "globally" through the
// lexing/parsing process.
class ParseAndLexContext {
 public:
  // Creates an instance analyzing the given input file.
  ParseAndLexContext(const std::string& input_file)
      : input_file_name(input_file) {}

  // Writes a syntax error diagnostic, containing message, for the input file at
  // the given line, to standard error.
  auto PrintDiagnostic(const std::string& message, int line_number) -> void;

  // The source range of the token being (or just) lex'd.
  yy::location current_token_position;

 private:
  // A path to the file processed, relative to the current working directory
  // when *this is called.
  const std::string input_file_name;
};

}  // namespace Carbon

// Gives flex the yylex prototype we want.
#define YY_DECL \
  yy::parser::symbol_type yylex(Carbon::ParseAndLexContext& context)

// Declares yylex for the parser's sake.
YY_DECL;

#endif  // EXECUTABLE_SYNTAX_DRIVER_H_
