// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>

#include "diagnostics/diagnostic_emitter.h"
#include "lexer/numeric_literal.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon {

// NOLINTNEXTLINE: Match the documented fuzzer entry point declaration style.
extern "C" int LLVMFuzzerTestOneInput(const unsigned char* data,
                                      std::size_t size) {
  auto token = NumericLiteralToken::Lex(
      llvm::StringRef(reinterpret_cast<const char*>(data), size));
  if (!token) {
    // Lexically not a numeric literal.
    return 0;
  }

  NumericLiteralToken::Parser parser(NullDiagnosticEmitter(), *token);
  if (parser.Check() == NumericLiteralToken::Parser::UnrecoverableError) {
    // Lexically OK, but token is meaningless.
    return 0;
  }

  // Ensure we can exercise the various queries on a valid literal.
  parser.GetRadix();
  (void)parser.GetMantissa();
  (void)parser.GetExponent();

  return 0;
}

}  // namespace Carbon
