// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_SYNTAX_PARSE_H_
#define EXECUTABLE_SEMANTICS_SYNTAX_PARSE_H_

#include <string>
#include <variant>

#include "executable_semantics/ast/ast.h"
#include "executable_semantics/common/arena.h"

namespace Carbon {

// This is the code given us by Bison, for now.
using SyntaxErrorCode = int;

// Returns the AST representing the contents of the named file, or an error code
// if parsing fails. Allocations go into the provided arena.
auto Parse(Nonnull<Arena*> arena, std::string_view input_file_name, bool trace)
    -> std::variant<Carbon::AST, SyntaxErrorCode>;

// Equivalent to `Parse`, but parses the contents of `file_contents`.
// `input_file_name` is used only for reporting source locations, and does
// not need to name a real file.
auto ParseFromString(Nonnull<Arena*> arena, std::string_view input_file_name,
                     std::string_view file_contents, bool trace)
    -> std::variant<Carbon::AST, SyntaxErrorCode>;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_SYNTAX_PARSE_H_
