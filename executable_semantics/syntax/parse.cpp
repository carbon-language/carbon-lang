// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/syntax/parse.h"

#include <iostream>

#include "executable_semantics/syntax/parse_and_lex_context.h"
#include "executable_semantics/syntax/parser.h"
#include "executable_semantics/tracing_flag.h"

extern FILE* yyin;

namespace Carbon {

// Returns an abstract representation of the program contained in the
// well-formed input file, or if the file was malformed, a description of the
// problem.
auto parse(const std::string& input_File_Name)
    -> std::variant<AST, SyntaxErrorCode> {
  yyin = fopen(input_File_Name.c_str(), "r");
  if (yyin == nullptr) {
    std::cerr << "Error opening '" << input_File_Name
              << "': " << std::strerror(errno) << std::endl;
    exit(1);
  }

  std::optional<AST> parsed_Input = std::nullopt;
  ParseAndLexContext context(input_File_Name);

  auto syntax_Error_Code = yy::parser(parsed_Input, context)();
  if (syntax_Error_Code != 0) {
    return syntax_Error_Code;
  }

  if (parsed_Input == std::nullopt) {
    std::cerr << "Internal error: parser validated syntax yet didn't produce "
                 "an AST.\n";
    exit(1);
  }
  return *parsed_Input;
}

}  // namespace Carbon
