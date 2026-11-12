// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEX_NUMERIC_LITERAL_H_
#define CARBON_TOOLCHAIN_LEX_NUMERIC_LITERAL_H_

#include <array>
#include <cstdint>
#include <type_traits>
#include <variant>

#include "common/check.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/base/int.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/base/value_ids.h"
#include "toolchain/diagnostics/emitter.h"
#include "toolchain/lex/character_set.h"
#include "toolchain/lex/source_loc.h"

namespace Carbon::Lex {

namespace Internal {

// Parses a single chunk of up to 19 decimal digits, which always fits in a
// `uint64_t`.
inline auto ParseDecimalChunk(llvm::StringRef digits) -> uint64_t {
  uint64_t chunk_val = 0;
  for (char c : digits) {
    chunk_val = chunk_val * 10 + (c - '0');
  }
  return chunk_val;
}

}  // namespace Internal

// A numeric literal token that has been extracted from a source buffer.
class NumericLiteral {
 public:
  struct UnrecoverableError {};

  // The ID of a numeric literal's value in the shared value stores.
  using ValueId = std::variant<IntId, RealId, UnrecoverableError>;

  // Lexes the numeric literal at the start of `source_text`, which must start
  // with a decimal digit.
  //
  // The supplied `source_text` must outlive the return value.
  //
  // `can_form_real_literal` returns whether the literal can be a real literal,
  // and is only called when that matters, so that callers don't pay to compute
  // it in the common case.
  //
  // The most common numeric literal, a single digit, is lexed inline so that it
  // doesn't pay for a call, and everything else out of line.
  template <typename CanFormRealLiteralFn>
    requires std::is_invocable_r_v<bool, CanFormRealLiteralFn>
  static auto Lex(llvm::StringRef source_text,
                  CanFormRealLiteralFn can_form_real_literal)
      -> NumericLiteral {
    CARBON_DCHECK(!source_text.empty() && IsDecimalDigit(source_text[0]),
                  "Numeric literal must start with a decimal digit");
    // A digit followed by the end of the text or by a byte that can't continue
    // a numeric literal is a complete one, such as the `1` in `x + 1`.
    if (source_text.size() == 1 || !IsContinuationByte(source_text[1]))
        [[likely]] {
      // Unlike `take_front(1)`, this doesn't clamp the size, so the compiler
      // can see that it's exactly one.
      return NumericLiteral(llvm::StringRef(source_text.begin(), 1));
    }
    return LexOutOfLine(source_text, can_form_real_literal());
  }

  // Computes the value of the token, adds it to `value_stores`, and returns its
  // ID. Emits diagnostics to the given emitter if the token is not valid.
  //
  // The most common numeric literal, a single digit, is handled inline so that
  // it doesn't pay for a call. Other decimal integers of up to 18 digits
  // without a leading zero, which always fit in an `int64_t`, bypass the
  // parser, and only build an `APInt` if they're too large to embed in their
  // ID.
  auto AddValue(SharedValueStores& value_stores,
                Diagnostics::Emitter<SourceLoc>& emitter) const -> ValueId {
    if (text_.size() == 1) [[likely]] {
      return value_stores.ints().Add(
          int64_t{Internal::ParseDecimalChunk(text_))};
    }
    return AddValueOutOfLine(text_, value_stores, emitter);
  }

  // Get the text corresponding to this literal.
  auto text() const -> llvm::StringRef { return text_; }

 private:
  class Parser;

  // Constructs a literal with the given text, which must be non-empty.
  explicit NumericLiteral(llvm::StringRef text) : text_(text) {}

  // The out-of-line parts of `Lex` and `AddValue`, which handle everything
  // other than a single digit. These are static, with text passed by value,
  // so that the inline paths' literals never need an address.
  static auto LexOutOfLine(llvm::StringRef source_text,
                           bool can_form_real_literal) -> NumericLiteral;
  static auto AddValueOutOfLine(llvm::StringRef text,
                                SharedValueStores& value_stores,
                                Diagnostics::Emitter<SourceLoc>& emitter)
      -> ValueId;

  // Returns whether `c` can continue a numeric literal that starts with a
  // sequence of decimal digits. If the byte after those digits can't, the
  // digits are a complete decimal integer literal.
  static auto IsContinuationByte(char c) -> bool {
    return ContinuationByteTable[uint8_t{c}];
  }

  // A table of the bytes that can continue a numeric literal after a sequence
  // of decimal digits: those that `Lex` always consumes (`[0-9A-Za-z_$]`), and
  // `.`, which can start the fractional part of a real literal.
  static constexpr std::array<bool, 256> ContinuationByteTable = [] {
    std::array<bool, 256> table = {};
    for (int c = '0'; c <= '9'; ++c) {
      table[c] = true;
    }
    for (int c = 'a'; c <= 'z'; ++c) {
      table[c] = true;
    }
    for (int c = 'A'; c <= 'Z'; ++c) {
      table[c] = true;
    }
    table['_'] = true;
    table['$'] = true;
    table['.'] = true;
    return table;
  }();

  // The text of the token.
  llvm::StringRef text_;
};

}  // namespace Carbon::Lex

#endif  // CARBON_TOOLCHAIN_LEX_NUMERIC_LITERAL_H_
