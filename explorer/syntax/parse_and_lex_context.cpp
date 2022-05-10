// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/syntax/parse_and_lex_context.h"

namespace Carbon {

auto ParseAndLexContext::RecordSyntaxError(const std::string& message,
                                           bool prefix_with_newline)
    -> Parser::symbol_type {
  // Optionally adds a newline in trace mode because trace prints an incomplete
  // line "Reading a token: " which can prevent LIT from finding expected
  // patterns.
  // TODO: support formatting of `SourceLocation` instances with formatv().
  std::string full_message;
  llvm::raw_string_ostream(full_message)
      << (prefix_with_newline && trace() ? "\n" : "")
      << "COMPILATION ERROR: " << source_loc() << ": " << message;
  error_messages_.push_back(full_message);

  // TODO: use `YYerror` token once bison is upgraded to at least 3.5.
  return Parser::make_END_OF_FILE(current_token_position);
}

}  // namespace Carbon
