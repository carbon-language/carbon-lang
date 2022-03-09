// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/syntax/parse_and_lex_context.h"

namespace Carbon {

auto ParseAndLexContext::RecordError(const std::string& message) -> void {
  error_messages_.push_back(message);
}

auto ParseAndLexContext::RecordLexerError(const std::string& message)
    -> Parser::symbol_type {
  // Adds a newline in trace mode because trace prints an incomplete line
  // "Reading a token: " which can prevent LIT from finding expected patterns.
  std::string full_message;
  llvm::raw_string_ostream(full_message)
      << (trace() ? "\n" : "") << "COMPILATION ERROR: " << source_loc() << ": "
      << message;
  RecordError(full_message);

  // TODO: use `YYerror` token once bison is upgraded to at least 3.5.
  return Parser::make_END_OF_FILE(current_token_position);
}

}  // namespace Carbon
