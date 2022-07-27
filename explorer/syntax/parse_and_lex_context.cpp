// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/syntax/parse_and_lex_context.h"

#include "explorer/common/error_builders.h"

namespace Carbon {

auto ParseAndLexContext::RecordSyntaxError(Error error) -> Parser::symbol_type {
  errors_.push_back(std::move(error));

  // TODO: use `YYerror` token once bison is upgraded to at least 3.5.
  return Parser::make_END_OF_FILE(current_token_position);
}

auto ParseAndLexContext::RecordSyntaxError(const std::string& message)
    -> Parser::symbol_type {
  return RecordSyntaxError(CompilationError(source_loc()) << message);
}

}  // namespace Carbon
