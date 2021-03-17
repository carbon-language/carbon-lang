// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>

#include "diagnostics/diagnostic_emitter.h"
#include "lexer/string_literal.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon {

// NOLINTNEXTLINE: Match the documented fuzzer entry point declaration style.
extern "C" int LLVMFuzzerTestOneInput(const unsigned char* data,
                                      std::size_t size) {
  auto token = LexedStringLiteral::Lex(
      llvm::StringRef(reinterpret_cast<const char*>(data), size));
  if (!token) {
    // Lexically not a string literal.
    return 0;
  }

  // Check multiline flag was computed correctly.
  if (token->IsMultiLine() != token->Text().contains('\n')) {
    __builtin_trap();
  }

  volatile auto value = token->ComputeValue(NullDiagnosticEmitter());
  (void)value;

  return 0;
}

}  // namespace Carbon
