// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LEXER_NUMERIC_LITERAL_H_
#define LEXER_NUMERIC_LITERAL_H_

#include <utility>

#include "diagnostics/diagnostic_emitter.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon {

// A numeric literal token that has been extracted from a source buffer.
class NumericLiteralToken {
 public:
  // Get the text corresponding to this literal.
  llvm::StringRef Text() const { return text; }

  // Extract a numeric literal from the given text, if it has a suitable form.
  //
  // The supplied `source_text` must outlive the return value.
  static auto Lex(llvm::StringRef source_text)
      -> llvm::Optional<NumericLiteralToken>;

  class Parser;

 private:
  NumericLiteralToken() {}

  // The text of the token.
  llvm::StringRef text;

  // The offset of the '.'. Set to text.size() if none is present.
  int radix_point;

  // The offset of the alphabetical character introducing the exponent. In a
  // valid literal, this will be an 'e' or a 'p', and may be followed by a '+'
  // or a '-', but for error recovery, this may simply be the last lowercase
  // letter in the invalid token. Always greater than or equal to radix_point.
  // Set to text.size() if none is present.
  int exponent;
};

// Parser for numeric literal tokens.
//
// Responsible for checking that a numeric literal is valid and meaningful and
// either diagnosing or extracting its meaning.
class NumericLiteralToken::Parser {
 public:
  Parser(DiagnosticEmitter& emitter, NumericLiteralToken literal);

  auto IsInteger() -> bool {
    return literal.radix_point == static_cast<int>(literal.text.size());
  }

  enum CheckResult {
    // The token is valid.
    Valid,
    // The token is invalid, but we've diagnosed and recovered from the error.
    RecoverableError,
    // The token is invalid, and we've diagnosed, but we can't assign meaning
    // to it.
    UnrecoverableError,
  };

  // Check that the numeric literal token is syntactically valid and
  // meaningful, and diagnose if not.
  auto Check() -> CheckResult;

  // Get the radix of this token. One of 2, 10, or 16.
  auto GetRadix() -> int { return radix; }

  // Get the mantissa of this token's value.
  auto GetMantissa() -> llvm::APInt;

  // Get the exponent of this token's value. This is always zero for an integer
  // literal.
  auto GetExponent() -> llvm::APInt;

 private:
  struct CheckDigitSequenceResult {
    bool ok;
    bool has_digit_separators = false;
  };

  auto CheckDigitSequence(llvm::StringRef text, int radix,
                          bool allow_digit_separators = true)
      -> CheckDigitSequenceResult;
  auto CheckDigitSeparatorPlacement(llvm::StringRef text, int radix,
                                    int num_digit_separators)
      -> bool;
  auto CheckLeadingZero() -> bool;
  auto CheckIntPart() -> bool;
  auto CheckFractionalPart() -> bool;
  auto CheckExponentPart() -> bool;

 private:
  DiagnosticEmitter& emitter;
  NumericLiteralToken literal;

  // The radix of the literal: 2, 10, or 16, for a prefix of '0b', no prefix,
  // or '0x', respectively.
  int radix = 10;

  // The various components of a numeric literal:
  //
  //     [radix] int_part [. fract_part [[ep] [+-] exponent_part]]
  llvm::StringRef int_part;
  llvm::StringRef fract_part;
  llvm::StringRef exponent_part;

  // Do we need to remove any special characters (digit separator or radix
  // point) before interpreting the mantissa or exponent as an integer?
  bool mantissa_needs_cleaning = false;
  bool exponent_needs_cleaning = false;

  // True if we found a `-` before `exponent_part`.
  bool exponent_is_negative = false;

  // True if we produced an error but recovered.
  bool recovered_from_error = false;
};

}  // namespace Carbon

#endif  // LEXER_NUMERIC_LITERAL_H_
