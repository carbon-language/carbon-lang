// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "llvm/ADT/StringRef.h"
#include "testing/fuzzing/libfuzzer.h"
#include "toolchain/diagnostics/null_diagnostics.h"
#include "toolchain/lex/numeric_literal.h"

namespace Carbon::Testing {

// NOLINTNEXTLINE: Match the documented fuzzer entry point declaration style.
extern "C" int LLVMFuzzerTestOneInput(const unsigned char* data, size_t size) {
  auto token = Lex::NumericLiteral::Lex(
      llvm::StringRef(reinterpret_cast<const char*>(data), size), true);
  if (!token) {
    // Lexically not a numeric literal.
    return 0;
  }

  volatile auto value =
      token->ComputeValue(Diagnostics::NullEmitter<const char*>());
  (void)value;
  return 0;
}

}  // namespace Carbon::Testing
