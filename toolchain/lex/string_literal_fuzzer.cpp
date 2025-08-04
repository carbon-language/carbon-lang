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
  auto literal = Lex::StringLiteral::Lex(
      llvm::StringRef(reinterpret_cast<const char*>(data), size));
  if (!literal) {
    // Lexically not a string literal.
    return 0;
  }

  if (!literal->is_terminated()) {
    // Found errors while parsing.
    return 0;
  }

  fprintf(stderr, "valid: %d\n", literal->is_terminated());
  fprintf(stderr, "size: %lu\n", literal->text().size());
  fprintf(stderr, "text: %s\n", literal->text().str().c_str());

  // Check multiline flag was computed correctly.
  switch (literal->kind()) {
    case Lex::StringLiteral::Kind::Char:
      break;

    case Lex::StringLiteral::Kind::SingleLine:
      CARBON_CHECK(!literal->text().contains('\n'));
      break;

    case Lex::StringLiteral::Kind::MultiLine:
    case Lex::StringLiteral::Kind::MultiLineWithDoubleQuotes:
      CARBON_CHECK(literal->text().contains('\n'));
      break;
  }

  auto* null_emitter = &Diagnostics::NullEmitter<const char*>();
  if (literal->kind() == Lex::StringLiteral::Kind::Char) {
    volatile auto value = literal->ComputeCharValue(*null_emitter);
    (void)value;
  } else {
    llvm::BumpPtrAllocator allocator;
    volatile auto value = literal->ComputeStringValue(allocator, *null_emitter);
    (void)value;
  }

  return 0;
}

}  // namespace Carbon::Testing
