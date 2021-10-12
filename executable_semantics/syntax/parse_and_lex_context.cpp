// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/syntax/parse_and_lex_context.h"

namespace Carbon {

auto ParseAndLexContext::PrintDiagnostic(const std::string& message) -> void {
  // TODO: Do we really want this to be fatal?  It makes the comment and the
  // name a lie, and renders some of the other yyparse() result propagation code
  // moot.
  FATAL_COMPILATION_ERROR(source_loc()) << message;
}

}  // namespace Carbon
