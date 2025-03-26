// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEX_NUMERIC_LITERAL_H_
#define CARBON_TOOLCHAIN_LEX_NUMERIC_LITERAL_H_

#include <optional>
#include <variant>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon::Lex {

// A numeric literal token that has been extracted from a source buffer.
class NumericLiteral {
 public:
  enum class Radix : int8_t { Binary = 2, Decimal = 10, Hexadecimal = 16 };

  // Value of an integer literal.
  struct IntValue {
    // An unsigned literal value.
    llvm::APInt value;
  };

  // Value of a real literal.
  struct RealValue {
    // The radix of the exponent, either Binary or Decimal.
    Radix radix;
    // The mantissa, represented as a variable-width unsigned integer.
    llvm::APInt mantissa;
    // The exponent, represented as a variable-width signed integer.
    llvm::APInt exponent;
  };

  struct UnrecoverableError {};

  using Value = std::variant<IntValue, RealValue, UnrecoverableError>;

  // Extract a numeric literal from the given text, if it has a suitable form.
  //
  // The supplied `source_text` must outlive the return value.
  static auto Lex(llvm::StringRef source_text, bool can_form_real_literal)
      -> std::optional<NumericLiteral>;

  // Compute the value of the token, if possible. Emit diagnostics to the given
  // emitter if the token is not valid.
  auto ComputeValue(Diagnostics::Emitter<const char*>& emitter) const -> Value;

  // Get the text corresponding to this literal.
  auto text() const -> llvm::StringRef { return text_; }

 private:
  class Parser;

  NumericLiteral() = default;

  // The text of the token.
  llvm::StringRef text_;

  // The offset of the '.'. Set to text.size() if none is present.
  int radix_point_;

  // The offset of the alphabetical character introducing the exponent. In a
  // valid literal, this will be an 'e' or a 'p', and may be followed by a '+'
  // or a '-', but for error recovery, this may simply be the last lowercase
  // letter in the invalid token. Always greater than or equal to radix_point.
  // Set to text.size() if none is present.
  int exponent_;
};

}  // namespace Carbon::Lex

#endif  // CARBON_TOOLCHAIN_LEX_NUMERIC_LITERAL_H_
