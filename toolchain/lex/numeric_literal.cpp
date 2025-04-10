// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/numeric_literal.h"

#include <bitset>

#include "common/check.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadicDetails.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/lex/character_set.h"
#include "toolchain/lex/helpers.h"

namespace Carbon::Lex {

auto NumericLiteral::Lex(llvm::StringRef source_text,
                         bool can_form_real_literal)
    -> std::optional<NumericLiteral> {
  NumericLiteral result;

  if (source_text.empty() || !IsDecimalDigit(source_text.front())) {
    return std::nullopt;
  }

  bool seen_plus_minus = false;
  bool seen_radix_point = false;
  bool seen_potential_exponent = false;

  // Greedily consume all following characters that might be part of a numeric
  // literal. This allows us to produce better diagnostics on invalid literals.
  //
  // TODO(zygoloid): Update lexical rules to specify that a numeric literal
  // cannot be immediately followed by an alphanumeric character.
  int i = 1;
  int n = source_text.size();
  for (; i != n; ++i) {
    char c = source_text[i];
    if (IsAlnum(c) || c == '_') {
      if (IsLower(c) && seen_radix_point && !seen_plus_minus) {
        result.exponent_ = i;
        seen_potential_exponent = true;
      }
      continue;
    }

    // Exactly one `.` can be part of the literal, but only if it's followed by
    // an alphanumeric character.
    if (c == '.' && can_form_real_literal && i + 1 != n &&
        IsAlnum(source_text[i + 1]) && !seen_radix_point) {
      result.radix_point_ = i;
      seen_radix_point = true;
      continue;
    }

    // A `+` or `-` continues the literal only if it's preceded by a lowercase
    // letter (which will be 'e' or 'p' or part of an invalid literal) and
    // followed by an alphanumeric character. This '+' or '-' cannot be an
    // operator because a literal cannot end in a lowercase letter.
    if ((c == '+' || c == '-') && seen_potential_exponent &&
        result.exponent_ == i - 1 && i + 1 != n &&
        IsAlnum(source_text[i + 1])) {
      // This is not possible because we don't update result.exponent after we
      // see a '+' or '-'.
      CARBON_CHECK(!seen_plus_minus, "should only consume one + or -");
      seen_plus_minus = true;
      continue;
    }

    break;
  }

  result.text_ = source_text.substr(0, i);
  if (!seen_radix_point) {
    result.radix_point_ = i;
  }
  if (!seen_potential_exponent) {
    result.exponent_ = i;
  }

  return result;
}

// Parser for numeric literal tokens.
//
// Responsible for checking that a numeric literal is valid and meaningful and
// either diagnosing or extracting its meaning.
class NumericLiteral::Parser {
 public:
  Parser(Diagnostics::Emitter<const char*>& emitter, NumericLiteral literal);

  auto IsInt() -> bool {
    return literal_.radix_point_ == static_cast<int>(literal_.text_.size());
  }

  // Check that the numeric literal token is syntactically valid and
  // meaningful, and diagnose if not. Returns `true` if the token was
  // sufficiently valid that we could determine its meaning. If `false` is
  // returned, a diagnostic has already been issued.
  auto Check() -> bool;

  // Get the radix of this token. One of 2, 10, or 16.
  auto GetRadix() -> Radix { return radix_; }

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

  auto CheckDigitSequence(llvm::StringRef text, Radix radix,
                          bool allow_digit_separators = true)
      -> CheckDigitSequenceResult;
  auto CheckLeadingZero() -> bool;
  auto CheckIntPart() -> bool;
  auto CheckFractionalPart() -> bool;
  auto CheckExponentPart() -> bool;

  Diagnostics::Emitter<const char*>& emitter_;
  NumericLiteral literal_;

  // The radix of the literal: 2, 10, or 16, for a prefix of '0b', no prefix,
  // or '0x', respectively.
  Radix radix_ = Radix::Decimal;

  // The various components of a numeric literal:
  //
  //     [radix] int_part [. fract_part [[ep] [+-] exponent_part]]
  llvm::StringRef int_part_;
  llvm::StringRef fract_part_;
  llvm::StringRef exponent_part_;

  // Do we need to remove any special characters (digit separator or radix
  // point) before interpreting the mantissa or exponent as an integer?
  bool mantissa_needs_cleaning_ = false;
  bool exponent_needs_cleaning_ = false;

  // True if we found a `-` before `exponent_part`.
  bool exponent_is_negative_ = false;
};

NumericLiteral::Parser::Parser(Diagnostics::Emitter<const char*>& emitter,
                               NumericLiteral literal)
    : emitter_(emitter), literal_(literal) {
  int_part_ = literal.text_.substr(0, literal.radix_point_);
  if (int_part_.consume_front("0x")) {
    radix_ = Radix::Hexadecimal;
  } else if (int_part_.consume_front("0b")) {
    radix_ = Radix::Binary;
  }

  fract_part_ = literal.text_.substr(
      literal.radix_point_ + 1, literal.exponent_ - literal.radix_point_ - 1);

  exponent_part_ = literal.text_.substr(literal.exponent_ + 1);
  if (!exponent_part_.consume_front("+")) {
    exponent_is_negative_ = exponent_part_.consume_front("-");
  }
}

// Check that the numeric literal token is syntactically valid and meaningful,
// and diagnose if not.
auto NumericLiteral::Parser::Check() -> bool {
  return CheckLeadingZero() && CheckIntPart() && CheckFractionalPart() &&
         CheckExponentPart();
}

// Parse a string that is known to be a valid base-radix integer into an
// APInt.  If needs_cleaning is true, the string may additionally contain '_'
// and '.' characters that should be ignored.
//
// Ignoring '.' is used when parsing a real literal. For example, when
// parsing 123.456e7, we want to decompose it into an integer mantissa
// (123456) and an exponent (7 - 3 = 4), and this routine is given the
// "123.456" to parse as the mantissa.
static auto ParseInt(llvm::StringRef digits, NumericLiteral::Radix radix,
                     bool needs_cleaning) -> llvm::APInt {
  llvm::SmallString<32> cleaned;
  if (needs_cleaning) {
    cleaned.reserve(digits.size());
    llvm::copy_if(digits, std::back_inserter(cleaned),
                  [](char c) { return c != '_' && c != '.'; });
    digits = cleaned;
  }

  llvm::APInt value;
  if (digits.getAsInteger(static_cast<int>(radix), value)) {
    llvm_unreachable("should never fail");
  }
  return value;
}

auto NumericLiteral::Parser::GetMantissa() -> llvm::APInt {
  const char* end = IsInt() ? int_part_.end() : fract_part_.end();
  llvm::StringRef digits(int_part_.begin(), end - int_part_.begin());
  return ParseInt(digits, radix_, mantissa_needs_cleaning_);
}

auto NumericLiteral::Parser::GetExponent() -> llvm::APInt {
  // Compute the effective exponent from the specified exponent, if any,
  // and the position of the radix point.
  llvm::APInt exponent(64, 0);
  if (!exponent_part_.empty()) {
    exponent =
        ParseInt(exponent_part_, Radix::Decimal, exponent_needs_cleaning_);

    // The exponent is a signed integer, and the number we just parsed is
    // non-negative, so ensure we have a wide enough representation to
    // include a sign bit. Also make sure the exponent isn't too narrow so
    // the calculation below can't lose information through overflow.
    if (exponent.isSignBitSet() || exponent.getBitWidth() < 64) {
      exponent = exponent.zext(std::max(64U, exponent.getBitWidth() + 1));
    }
    if (exponent_is_negative_) {
      exponent.negate();
    }
  }

  // Each character after the decimal point reduces the effective exponent.
  int excess_exponent = fract_part_.size();
  if (radix_ == Radix::Hexadecimal) {
    excess_exponent *= 4;
  }
  exponent -= excess_exponent;
  if (exponent_is_negative_ && !exponent.isNegative()) {
    // We overflowed. Note that we can only overflow by a little, and only
    // from negative to positive, because exponent is at least 64 bits wide
    // and excess_exponent is bounded above by four times the size of the
    // input buffer, which we assume fits into 32 bits.
    exponent = exponent.zext(exponent.getBitWidth() + 1);
    exponent.setSignBit();
  }
  return exponent;
}

// Check that a digit sequence is valid: that it contains one or more digits,
// contains only digits in the specified base, and that any digit separators
// are present and correctly positioned.
auto NumericLiteral::Parser::CheckDigitSequence(llvm::StringRef text,
                                                Radix radix,
                                                bool allow_digit_separators)
    -> CheckDigitSequenceResult {
  std::bitset<256> valid_digits;
  switch (radix) {
    case Radix::Binary:
      for (char c : "01") {
        valid_digits[static_cast<unsigned char>(c)] = true;
      }
      break;
    case Radix::Decimal:
      for (char c : "0123456789") {
        valid_digits[static_cast<unsigned char>(c)] = true;
      }
      break;
    case Radix::Hexadecimal:
      for (char c : "0123456789ABCDEF") {
        valid_digits[static_cast<unsigned char>(c)] = true;
      }
      break;
  }

  int num_digit_separators = 0;

  for (int i = 0, n = text.size(); i != n; ++i) {
    char c = text[i];
    if (valid_digits[static_cast<unsigned char>(c)]) {
      continue;
    }

    if (c == '_') {
      // A digit separator cannot appear at the start of a digit sequence,
      // next to another digit separator, or at the end.
      if (!allow_digit_separators || i == 0 || text[i - 1] == '_' ||
          i + 1 == n) {
        CARBON_DIAGNOSTIC(InvalidDigitSeparator, Error,
                          "misplaced digit separator in numeric literal");
        emitter_.Emit(text.begin() + 1, InvalidDigitSeparator);
      }
      ++num_digit_separators;
      continue;
    }

    CARBON_DIAGNOSTIC(
        InvalidDigit, Error,
        "invalid digit '{0}' in {1:=2:binary|=10:decimal|=16:hexadecimal} "
        "numeric literal",
        char, Diagnostics::IntAsSelect);
    emitter_.Emit(text.begin() + i, InvalidDigit, c, static_cast<int>(radix));
    return {.ok = false};
  }

  if (num_digit_separators == static_cast<int>(text.size())) {
    CARBON_DIAGNOSTIC(EmptyDigitSequence, Error,
                      "empty digit sequence in numeric literal");
    emitter_.Emit(text.begin(), EmptyDigitSequence);
    return {.ok = false};
  }

  if (!CanLexInt(emitter_, text)) {
    return {.ok = false};
  }

  return {.ok = true, .has_digit_separators = (num_digit_separators != 0)};
}

// Check that we don't have a '0' prefix on a non-zero decimal integer.
auto NumericLiteral::Parser::CheckLeadingZero() -> bool {
  if (radix_ == Radix::Decimal && int_part_.starts_with("0") &&
      int_part_ != "0") {
    CARBON_DIAGNOSTIC(UnknownBaseSpecifier, Error,
                      "unknown base specifier in numeric literal");
    emitter_.Emit(int_part_.begin(), UnknownBaseSpecifier);
    return false;
  }
  return true;
}

// Check the integer part (before the '.', if any) is valid.
auto NumericLiteral::Parser::CheckIntPart() -> bool {
  auto int_result = CheckDigitSequence(int_part_, radix_);
  mantissa_needs_cleaning_ |= int_result.has_digit_separators;
  return int_result.ok;
}

// Check the fractional part (after the '.' and before the exponent, if any)
// is valid.
auto NumericLiteral::Parser::CheckFractionalPart() -> bool {
  if (IsInt()) {
    return true;
  }

  if (radix_ == Radix::Binary) {
    CARBON_DIAGNOSTIC(BinaryRealLiteral, Error,
                      "binary real number literals are not supported");
    emitter_.Emit(literal_.text_.begin() + literal_.radix_point_,
                  BinaryRealLiteral);
    // Carry on and parse the binary real literal anyway.
  }

  // We need to remove a '.' from the mantissa.
  mantissa_needs_cleaning_ = true;

  return CheckDigitSequence(fract_part_, radix_,
                            /*allow_digit_separators=*/false)
      .ok;
}

// Check the exponent part (if any) is valid.
auto NumericLiteral::Parser::CheckExponentPart() -> bool {
  if (literal_.exponent_ == static_cast<int>(literal_.text_.size())) {
    return true;
  }

  char expected_exponent_kind = (radix_ == Radix::Decimal ? 'e' : 'p');
  if (literal_.text_[literal_.exponent_] != expected_exponent_kind) {
    CARBON_DIAGNOSTIC(WrongRealLiteralExponent, Error,
                      "expected '{0}' to introduce exponent", char);
    emitter_.Emit(literal_.text_.begin() + literal_.exponent_,
                  WrongRealLiteralExponent, expected_exponent_kind);
    return false;
  }

  auto exponent_result = CheckDigitSequence(exponent_part_, Radix::Decimal);
  exponent_needs_cleaning_ = exponent_result.has_digit_separators;
  return exponent_result.ok;
}

// Parse the token and compute its value.
auto NumericLiteral::ComputeValue(
    Diagnostics::Emitter<const char*>& emitter) const -> Value {
  Parser parser(emitter, *this);

  if (!parser.Check()) {
    return UnrecoverableError();
  }

  if (parser.IsInt()) {
    return IntValue{.value = parser.GetMantissa()};
  }

  return RealValue{
      .radix = (parser.GetRadix() == Radix::Decimal ? Radix::Decimal
                                                    : Radix::Binary),
      .mantissa = parser.GetMantissa(),
      .exponent = parser.GetExponent()};
}

}  // namespace Carbon::Lex
