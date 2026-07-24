// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "common/check.h"
#include "testing/fuzzing/libfuzzer.h"
#include "toolchain/lex/mismatched_brackets.h"

namespace Carbon::Lex::Testing {
namespace {

struct Insertion {
  TokenIndex anchor;
  bool is_after;
  size_t order;
  TokenKind kind;
};

// Verifies that applying all corrections to the input token sequence results in
// a correctly bracket-balanced stream.
auto VerifyBracketBalance(llvm::ArrayRef<MismatchedBracketToken> tokens,
                          llvm::ArrayRef<BracketCorrection> corrections)
    -> void {
  llvm::SmallVector<bool> is_replaced_with_error(tokens.size(), false);
  llvm::SmallVector<Insertion> insertions;

  for (size_t order = 0; order < corrections.size(); ++order) {
    const auto& corr = corrections[order];
    if (corr.fix_action == BracketFixAction::ReplaceWithError) {
      is_replaced_with_error[corr.fix_token_index.index] = true;
    } else if (corr.fix_action == BracketFixAction::InsertBefore) {
      insertions.push_back({
          .anchor = corr.fix_token_index,
          .is_after = false,
          .order = order,
          .kind = corr.fix_token_kind,
      });
    } else if (corr.fix_action == BracketFixAction::InsertAfter) {
      insertions.push_back({
          .anchor = corr.fix_token_index,
          .is_after = true,
          .order = order,
          .kind = corr.fix_token_kind,
      });
    }
  }

  // This must match `ErrorRecoveryBuffer::Apply` in lex.cpp, which is how the
  // corrections are actually applied to the token stream. In particular, at a
  // shared insertion point, closing brackets are inserted before opening
  // brackets, so that closing an outer group and opening a new one land in
  // the correct order.
  llvm::stable_sort(insertions, [](const Insertion& a, const Insertion& b) {
    TokenIndex a_target =
        a.is_after ? TokenIndex(a.anchor.index + 1) : a.anchor;
    TokenIndex b_target =
        b.is_after ? TokenIndex(b.anchor.index + 1) : b.anchor;
    if (a_target != b_target) {
      return a_target < b_target;
    }
    if (a.is_after != b.is_after) {
      return a.is_after;
    }
    bool a_is_closing = a.kind.is_closing_symbol();
    bool b_is_closing = b.kind.is_closing_symbol();
    if (a_is_closing != b_is_closing) {
      return a_is_closing;
    }
    if (a.is_after) {
      return a.order < b.order;
    } else {
      return a.order > b.order;
    }
  });

  llvm::SmallVector<TokenKind> resulting_stream;
  size_t ins_idx = 0;

  for (int32_t i = 0; i <= static_cast<int32_t>(tokens.size()); ++i) {
    while (ins_idx < insertions.size()) {
      TokenIndex target = insertions[ins_idx].is_after
                              ? TokenIndex(insertions[ins_idx].anchor.index + 1)
                              : insertions[ins_idx].anchor;
      if (target.index != i) {
        break;
      }
      resulting_stream.push_back(insertions[ins_idx].kind);
      ++ins_idx;
    }

    if (i < static_cast<int32_t>(tokens.size())) {
      if (!is_replaced_with_error[i]) {
        resulting_stream.push_back(ToTokenKind(tokens[i].kind));
      }
    }
  }

  llvm::SmallVector<TokenKind> stack;
  for (TokenKind kind : resulting_stream) {
    if (kind.is_opening_symbol()) {
      stack.push_back(kind);
    } else if (kind.is_closing_symbol()) {
      CARBON_CHECK(!stack.empty(),
                   "Unmatched closing bracket in fixed stream!");
      TokenKind top = stack.pop_back_val();
      CARBON_CHECK(top.closing_symbol() == kind,
                   "Mismatched bracket pair in fixed stream!");
    }
  }
  CARBON_CHECK(stack.empty(),
               "Unclosed opening brackets remaining in fixed stream!");
}

}  // namespace

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

  tokens.push_back(MismatchedBracketToken{
      .token_index = TokenIndex(token_idx++),
      .kind = BracketTokenKind::FileEnd,
      .line = current_line,
      .line_indent = 0,
      .column = 1,
      .is_at_end_of_line = true,
      .is_struct_brace = false,
  });

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

  // Verification: applying fixes must result in a balanced bracket sequence.
  VerifyBracketBalance(tokens, corrections);

  return 0;
}

}  // namespace Carbon::Lex::Testing
