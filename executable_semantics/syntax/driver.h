// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Helpers should be added here when logic in syntax.ypp is more than a single
// statement. The intent is to minimize the amount of C++ in the .ypp file, to
// improve ease of maintenance.

#ifndef EXECUTABLE_SEMANTICS_SYNTAX_DRIVER_H_
#define EXECUTABLE_SEMANTICS_SYNTAX_DRIVER_H_

#include <variant>

#include "executable_semantics/syntax/driver.h"
#include "executable_semantics/syntax/parser.h"  // from parser.ypp

namespace Carbon {

// An encapsulation of the lexing/parsing process and all its state.
class SyntaxDriver {
 public:
  // Creates an instance analyzing the given input file.
  SyntaxDriver(const std::string& inputFile) : inputFileName(inputFile) {}

  // Indication of a syntax error.
  //
  // Will be used as an exit code, so the low 8 bits should never be zero.
  using Error = int;

  // Returns an abstract representation of the program contained in the
  // well-formed input file, or if the file was malformed, a description of the
  // syntax problem.
  auto operator()() -> std::variant<Carbon::AST, Error>;

  // Writes a syntax error diagnostic, containing message, for the input file at
  // the given line, to standard error.
  auto PrintDiagnostic(const char* message, int lineNumber) -> void;
  SyntaxDriver(const SyntaxDriver&) = delete;
  SyntaxDriver& operator=(const SyntaxDriver&) = delete;

 private:
  // A path to the file processed, relative to the current working directory
  // when *this is called.
  const std::string inputFileName;
};

}  // namespace Carbon

// Gives flex the yylex prototype we want.
#define YY_DECL int yylex(Carbon::SyntaxDriver& driver)

// Declares yylex for the parser's sake.
YY_DECL;

#endif  // EXECUTABLE_SYNTAX_DRIVER_EXEC_H_
