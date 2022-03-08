// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/syntax/parse_and_lex_context.h"

namespace Carbon {

auto ParseAndLexContext::RecordError(const std::string& message) -> void {
  error_messages_.push_back(message);
}

}  // namespace Carbon
