// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/syntax/prelude.h"

#include "explorer/syntax/parse.h"

namespace Carbon {

// Adds the Carbon prelude to `declarations`.
void AddPrelude(std::string_view prelude_file_name, Nonnull<Arena*> arena,
                std::vector<Nonnull<Declaration*>>* declarations) {
  ErrorOr<AST> parse_result = Parse(arena, prelude_file_name, false);
  if (!parse_result.ok()) {
    // Try again with tracing, to help diagnose the problem.
    ErrorOr<AST> trace_parse_result = Parse(arena, prelude_file_name, true);
    CARBON_FATAL() << "Failed to parse prelude:\n"
                   << trace_parse_result.error();
  }
  const auto& prelude = *parse_result;
  declarations->insert(declarations->begin(), prelude.declarations.begin(),
                       prelude.declarations.end());
}

}  // namespace Carbon
