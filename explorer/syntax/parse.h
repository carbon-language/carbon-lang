// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPLORER_SYNTAX_PARSE_H_
#define EXPLORER_SYNTAX_PARSE_H_

#include <string>
#include <variant>

#include "explorer/ast/ast.h"
#include "explorer/common/arena.h"

namespace Carbon {

// Returns the AST representing the contents of the named file, or an error code
// if parsing fails. Allocations go into the provided arena.
auto Parse(Nonnull<Arena*> arena, std::string_view input_file_name,
           bool parser_debug) -> ErrorOr<Carbon::AST>;

// Equivalent to `Parse`, but parses the contents of `file_contents`.
// `input_file_name` is used only for reporting source locations, and does
// not need to name a real file.
auto ParseFromString(Nonnull<Arena*> arena, std::string_view input_file_name,
                     std::string_view file_contents, bool parser_debug)
    -> ErrorOr<Carbon::AST>;

}  // namespace Carbon

#endif  // EXPLORER_SYNTAX_PARSE_H_
