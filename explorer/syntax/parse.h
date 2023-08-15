// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_SYNTAX_PARSE_H_
#define CARBON_EXPLORER_SYNTAX_PARSE_H_

#include <string>
#include <variant>

#include "explorer/ast/ast.h"
#include "explorer/base/arena.h"
#include "explorer/base/source_location.h"
#include "llvm/Support/VirtualFileSystem.h"

namespace Carbon {

// Returns the AST representing the contents of the named file, or an error code
// if parsing fails. Allocations go into the provided arena.
auto Parse(llvm::vfs::FileSystem& fs, Nonnull<Arena*> arena,
           std::string_view input_file_name, FileKind file_kind,
           bool parser_debug) -> ErrorOr<Carbon::AST>;

// Equivalent to `Parse`, but parses the contents of `file_contents`.
// `input_file_name` is used only for reporting source locations, and does
// not need to name a real file.
auto ParseFromString(Nonnull<Arena*> arena, std::string_view input_file_name,
                     FileKind file_kind, std::string_view file_contents,
                     bool parser_debug) -> ErrorOr<Carbon::AST>;

}  // namespace Carbon

#endif  // CARBON_EXPLORER_SYNTAX_PARSE_H_
