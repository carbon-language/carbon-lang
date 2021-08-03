// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_SYNTAX_PARSE_H_
#define EXECUTABLE_SEMANTICS_SYNTAX_PARSE_H_

#include <string>
#include <variant>

#include "executable_semantics/ast/abstract_syntax_tree.h"

namespace Carbon {

// This is the code given us by Bison, for now.
using SyntaxErrorCode = int;

// Returns the AST representing the contents of the named file, or an error code
// if parsing fails.
auto parse(const std::string& input_file_name)
    -> std::variant<Carbon::AST, SyntaxErrorCode>;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_SYNTAX_PARSE_H_
