// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_SYNTAX_ANTLR_PARSE_H_
#define CARBON_EXPLORER_SYNTAX_ANTLR_PARSE_H_

#include <string>

#include "common/error.h"
#include "explorer/ast/ast.h"
#include "explorer/common/arena.h"

namespace Carbon::Antlr {

// Returns the AST representing the contents of the named file, or an error code
// if parsing fails. Allocations go into the provided arena.
auto Parse(Nonnull<Arena*> arena, std::string_view input_file_name,
           bool parser_debug) -> ErrorOr<Carbon::AST>;

}  // namespace Carbon::Antlr

#endif  // CARBON_EXPLORER_SYNTAX_ANTLR_PARSE_H_
