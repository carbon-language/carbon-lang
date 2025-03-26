// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "common/check.h"
#include "llvm/ADT/StringRef.h"
#include "testing/fuzzing/libfuzzer.h"
#include "toolchain/diagnostics/null_diagnostics.h"
#include "toolchain/lex/string_literal.h"

namespace Carbon::Testing {

// NOLINTNEXTLINE: Match the documented fuzzer entry point declaration style.
extern "C" int LLVMFuzzerTestOneInput(const unsigned char* data, size_t size) {
  auto token = Lex::StringLiteral::Lex(
      llvm::StringRef(reinterpret_cast<const char*>(data), size));
  if (!token) {
    // Lexically not a string literal.
    return 0;
  }

  if (!token->is_terminated()) {
    // Found errors while parsing.
    return 0;
  }

  fprintf(stderr, "valid: %d\n", token->is_terminated());
  fprintf(stderr, "size: %lu\n", token->text().size());
  fprintf(stderr, "text: %s\n", token->text().str().c_str());

  // Check multiline flag was computed correctly.
  CARBON_CHECK(token->is_multi_line() == token->text().contains('\n'));

  llvm::BumpPtrAllocator allocator;
  volatile auto value =
      token->ComputeValue(allocator, Diagnostics::NullEmitter<const char*>());
  (void)value;

  return 0;
}

}  // namespace Carbon::Testing
