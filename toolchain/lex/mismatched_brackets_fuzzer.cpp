// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "common/check.h"
#include "testing/fuzzing/libfuzzer.h"
#include "toolchain/lex/mismatched_brackets.h"

namespace Carbon::Lex::Testing {

// Fuzz tester for mismatched bracket recovery.
// NOLINTNEXTLINE: Match the documented fuzzer entry point declaration style.
extern "C" int LLVMFuzzerTestOneInput(const unsigned char* data, size_t size) {
  if (size > 2000) {
    return 0;
  }

  llvm::SmallVector<MismatchedBracketToken> tokens;
  tokens.reserve(size / 4);

  size_t i = 0;
  int32_t token_idx = 0;
  int32_t current_line = 1;

  while (i + 4 <= size) {
    uint8_t kind_byte = data[i++];
    uint8_t indent_byte = data[i++];
    uint8_t flags_byte = data[i++];
    uint8_t line_delta = data[i++];

    auto kind = static_cast<BracketTokenKind>(kind_byte % 9);
    int32_t indent = (indent_byte % 32) * 2;
    bool is_eol = (flags_byte & 1) != 0;
    bool is_struct = (flags_byte & 2) != 0;
    current_line += (line_delta % 3);

    tokens.push_back(MismatchedBracketToken{
        .token_index = TokenIndex(token_idx++),
        .kind = kind,
        .line = current_line,
        .line_indent = indent,
        .column = indent + 1,
        .is_at_end_of_line = is_eol,
        .is_struct_brace = is_struct,
    });
  }

  auto corrections = FixMismatchedBrackets(tokens);

  // Invariant verification: all indices must be valid.
  for (const auto& corr : corrections) {
    CARBON_CHECK(corr.diagnostic_token_index.index >= 0 &&
                     corr.diagnostic_token_index.index < token_idx,
                 "Invalid diag token index!");
    CARBON_CHECK(corr.fix_token_index.index >= 0 &&
                     corr.fix_token_index.index < token_idx,
                 "Invalid fix token index!");
  }

  return 0;
}

}  // namespace Carbon::Lex::Testing
