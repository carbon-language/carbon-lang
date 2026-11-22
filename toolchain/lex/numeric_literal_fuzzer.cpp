// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "llvm/ADT/StringRef.h"
#include "testing/fuzzing/libfuzzer.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/diagnostics/null_diagnostics.h"
#include "toolchain/lex/character_set.h"
#include "toolchain/lex/numeric_literal.h"

namespace Carbon::Testing {

// NOLINTNEXTLINE: Match the documented fuzzer entry point declaration style.
extern "C" int LLVMFuzzerTestOneInput(const unsigned char* data, size_t size) {
  llvm::StringRef text(reinterpret_cast<const char*>(data), size);
  if (text.empty() || !Lex::IsDecimalDigit(text.front())) {
    // Lexically not a numeric literal.
    return 0;
  }

  SharedValueStores value_stores;
  Lex::NumericLiteral::Lex(text, [] {
    return true;
  }).AddValue(value_stores, Diagnostics::NullEmitter<Lex::SourceLoc>());
  return 0;
}

}  // namespace Carbon::Testing
