// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lexer/numeric_literal.h"

#include <bitset>

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/lexer/character_set.h"

namespace Carbon {

namespace {
struct EmptyDigitSequence : SimpleDiagnostic<EmptyDigitSequence> {
  static constexpr llvm::StringLiteral ShortName = "syntax-invalid-number";
  static constexpr llvm::StringLiteral Message =
      "Empty digit sequence in numeric literal.";
};

struct InvalidDigit {
  static constexpr llvm::StringLiteral ShortName = "syntax-invalid-number";

  char digit;
  int radix;

  auto Format() -> std::string {
    return llvm::formatv(
               "Invalid digit '{0}' in {1} numeric literal.", digit,
               (radix == 2 ? "binary"
                           : (radix == 16 ? "hexadecimal" : "decimal")))
        .str();
  }
};

struct InvalidDigitSeparator : SimpleDiagnostic<InvalidDigitSeparator> {
  static constexpr llvm::StringLiteral ShortName = "syntax-invalid-number";
  static constexpr llvm::StringLiteral Message =
      "Misplaced digit separator in numeric literal.";
};

struct IrregularDigitSeparators {
  static constexpr llvm::StringLiteral ShortName =
      "syntax-irregular-digit-separators";

  int radix;

  auto Format() -> std::string {
    assert((radix == 10 or radix == 16) and "unexpected radix");
    return llvm::formatv(
               "Digit separators in {0} number should appear every {1} "
               "characters from the right.",
               (radix == 10 ? "decimal" : "hexadecimal"),
               (radix == 10 ? "3" : "4"))
        .str();
  }
};

struct UnknownBaseSpecifier : SimpleDiagnostic<UnknownBaseSpecifier> {
  static constexpr llvm::StringLiteral ShortName = "syntax-invalid-number";
  static constexpr llvm::StringLiteral Message =
      "Unknown base specifier in numeric literal.";
};

struct BinaryRealLiteral : SimpleDiagnostic<BinaryRealLiteral> {
  static constexpr llvm::StringLiteral ShortName = "syntax-invalid-number";
  static constexpr llvm::StringLiteral Message =
      "Binary real number literals are not supported.";
};

struct WrongRealLiteralExponent {
  static constexpr llvm::StringLiteral ShortName = "syntax-invalid-number";

  char expected;

  auto Format() -> std::string {
    return llvm::formatv("Expected '{0}' to introduce exponent.", expected)
        .str();
  }
};
}  // namespace

auto LexedNumericLiteral::Lex(llvm::StringRef source_text)
    -> llvm::Optional<LexedNumericLiteral> {
  LexedNumericLiteral result;

  if (source_text.empty() or not IsDecimalDigit(source_text.front())) {
    return llvm::None;
  }

  bool seen_plus_minus = false;
  bool seen_radix_point = false;
  bool seen_potential_exponent = false;

  // Greedily consume all following characters that might be part of a numeric
  // literal. This allows us to produce better diagnostics on invalid literals.
  //
  // TODO(zygoloid): Update lexical rules to specify that a numeric literal
  // cannot be immediately followed by an alphanumeric character.
  int i = 1, n = source_text.size();
  for (; i != n; ++i) {
    char c = source_text[i];
    if (IsAlnum(c) or c == '_') {
      if (IsLower(c) and seen_radix_point and not seen_plus_minus) {
        result.exponent = i;
        seen_potential_exponent = true;
      }
      continue;
    }

    // Exactly one `.` can be part of the literal, but only if it's followed by
    // an alphanumeric character.
    if (c == '.' and i + 1 != n and IsAlnum(source_text[i + 1]) and
        not seen_radix_point) {
      result.radix_point = i;
      seen_radix_point = true;
      continue;
    }

    // A `+` or `-` continues the literal only if it's preceded by a lowercase
    // letter (which will be 'e' or 'p' or part of an invalid literal) and
    // followed by an alphanumeric character. This '+' or '-' cannot be an
    // operator because a literal cannot end in a lowercase letter.
    if ((c == '+' or c == '-') and seen_potential_exponent and
        result.exponent == i - 1 and i + 1 != n and IsAlnum(source_text[i + 1])) {
      // This is not possible because we don't update result.exponent after we
      // see a '+' or '-'.
      assert(not seen_plus_minus and "should only consume one + or -");
      seen_plus_minus = true;
      continue;
    }

    break;
  }

  result.text = source_text.substr(0, i);
  if (not seen_radix_point) {
    result.radix_point = i;
  }
  if (not seen_potential_exponent) {
    result.exponent = i;
  }

  return result;
}

// Parser for numeric literal tokens.
//
// Responsible for checking that a numeric literal is valid and meaningful and
// either diagnosing or extracting its meaning.
class LexedNumericLiteral::Parser {
 public:
  Parser(DiagnosticEmitter<const char*>& emitter, LexedNumericLiteral literal);

  auto IsInteger() -> bool {
    return literal.radix_point == static_cast<int>(literal.text.size());
  }

  // Check that the numeric literal token is syntactically valid and
  // meaningful, and diagnose if not. Returns `true` if the token was
  // sufficiently valid that we could determine its meaning. If `false` is
  // returned, a diagnostic has already been issued.
  auto Check() -> bool;

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
                                    int num_digit_separators) -> void;
  auto CheckLeadingZero() -> bool;
  auto CheckIntPart() -> bool;
  auto CheckFractionalPart() -> bool;
  auto CheckExponentPart() -> bool;

 private:
  DiagnosticEmitter<const char*>& emitter;
  LexedNumericLiteral literal;

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
};

LexedNumericLiteral::Parser::Parser(DiagnosticEmitter<const char*>& emitter,
                                    LexedNumericLiteral literal)
    : emitter(emitter), literal(literal) {
  int_part = literal.text.substr(0, literal.radix_point);
  if (int_part.consume_front("0x")) {
    radix = 16;
  } else if (int_part.consume_front("0b")) {
    radix = 2;
  }

  fract_part = literal.text.substr(literal.radix_point + 1,
                                   literal.exponent - literal.radix_point - 1);

  exponent_part = literal.text.substr(literal.exponent + 1);
  if (not exponent_part.consume_front("+")) {
    exponent_is_negative = exponent_part.consume_front("-");
  }
}

// Check that the numeric literal token is syntactically valid and meaningful,
// and diagnose if not.
auto LexedNumericLiteral::Parser::Check() -> bool {
  return CheckLeadingZero() and CheckIntPart() and CheckFractionalPart() and
         CheckExponentPart();
}

// Parse a string that is known to be a valid base-radix integer into an
// APInt.  If needs_cleaning is true, the string may additionally contain '_'
// and '.' characters that should be ignored.
//
// Ignoring '.' is used when parsing a real literal. For example, when
// parsing 123.456e7, we want to decompose it into an integer mantissa
// (123456) and an exponent (7 - 3 = 2), and this routine is given the
// "123.456" to parse as the mantissa.
static auto ParseInteger(llvm::StringRef digits, int radix, bool needs_cleaning)
    -> llvm::APInt {
  llvm::SmallString<32> cleaned;
  if (needs_cleaning) {
    cleaned.reserve(digits.size());
    std::remove_copy_if(digits.begin(), digits.end(),
                        std::back_inserter(cleaned),
                        [](char c) { return c == '_' or c == '.'; });
    digits = cleaned;
  }

  llvm::APInt value;
  if (digits.getAsInteger(radix, value)) {
    llvm_unreachable("should never fail");
  }
  return value;
}

auto LexedNumericLiteral::Parser::GetMantissa() -> llvm::APInt {
  const char* end = IsInteger() ? int_part.end() : fract_part.end();
  llvm::StringRef digits(int_part.begin(), end - int_part.begin());
  return ParseInteger(digits, radix, mantissa_needs_cleaning);
}

auto LexedNumericLiteral::Parser::GetExponent() -> llvm::APInt {
  // Compute the effective exponent from the specified exponent, if any,
  // and the position of the radix point.
  llvm::APInt exponent(64, 0);
  if (not exponent_part.empty()) {
    exponent = ParseInteger(exponent_part, 10, exponent_needs_cleaning);

    // The exponent is a signed integer, and the number we just parsed is
    // non-negative, so ensure we have a wide enough representation to
    // include a sign bit. Also make sure the exponent isn't too narrow so
    // the calculation below can't lose information through overflow.
    if (exponent.isSignBitSet() or exponent.getBitWidth() < 64) {
      exponent = exponent.zext(std::max(64u, exponent.getBitWidth() + 1));
    }
    if (exponent_is_negative) {
      exponent.negate();
    }
  }

  // Each character after the decimal point reduces the effective exponent.
  int excess_exponent = fract_part.size();
  if (radix == 16) {
    excess_exponent *= 4;
  }
  exponent -= excess_exponent;
  if (exponent_is_negative and not exponent.isNegative()) {
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
auto LexedNumericLiteral::Parser::CheckDigitSequence(
    llvm::StringRef text, int radix, bool allow_digit_separators)
    -> CheckDigitSequenceResult {
  assert((radix == 2 or radix == 10 or radix == 16) and "unknown radix");

  std::bitset<256> valid_digits;
  if (radix == 2) {
    for (char c : "01") {
      valid_digits[static_cast<unsigned char>(c)] = true;
    }
  } else if (radix == 10) {
    for (char c : "0123456789") {
      valid_digits[static_cast<unsigned char>(c)] = true;
    }
  } else {
    for (char c : "0123456789ABCDEF") {
      valid_digits[static_cast<unsigned char>(c)] = true;
    }
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
      if (not allow_digit_separators or i == 0 or text[i - 1] == '_' or
          i + 1 == n) {
        emitter.EmitError<InvalidDigitSeparator>(text.begin() + i);
      }
      ++num_digit_separators;
      continue;
    }

    emitter.EmitError<InvalidDigit>(text.begin() + i,
                                    {.digit = c, .radix = radix});
    return {.ok = false};
  }

  if (num_digit_separators == static_cast<int>(text.size())) {
    emitter.EmitError<EmptyDigitSequence>(text.begin());
    return {.ok = false};
  }

  // Check that digit separators occur in exactly the expected positions.
  if (num_digit_separators) {
    CheckDigitSeparatorPlacement(text, radix, num_digit_separators);
  }

  return {.ok = true, .has_digit_separators = (num_digit_separators != 0)};
}

// Given a number with digit separators, check that the digit separators are
// correctly positioned.
auto LexedNumericLiteral::Parser::CheckDigitSeparatorPlacement(
    llvm::StringRef text, int radix, int num_digit_separators) -> void {
  assert(std::count(text.begin(), text.end(), '_') == num_digit_separators and
         "given wrong number of digit separators");

  if (radix == 2) {
    // There are no restrictions on digit separator placement for binary
    // literals.
    return;
  }

  assert((radix == 10 or radix == 16) and
         "unexpected radix for digit separator checks");

  auto diagnose_irregular_digit_separators = [&]() {
    emitter.EmitError<IrregularDigitSeparators>(text.begin(), {.radix = radix});
  };

  // For decimal and hexadecimal digit sequences, digit separators must form
  // groups of 3 or 4 digits (4 or 5 characters), respectively.
  int stride = (radix == 10 ? 4 : 5);
  int remaining_digit_separators = num_digit_separators;
  auto pos = text.end();
  while (pos - text.begin() >= stride) {
    pos -= stride;
    if (*pos != '_') {
      diagnose_irregular_digit_separators();
      return;
    }

    --remaining_digit_separators;
  }

  // Check there weren't any other digit separators.
  if (remaining_digit_separators) {
    diagnose_irregular_digit_separators();
  }
};

// Check that we don't have a '0' prefix on a non-zero decimal integer.
auto LexedNumericLiteral::Parser::CheckLeadingZero() -> bool {
  if (radix == 10 and int_part.startswith("0") and int_part != "0") {
    emitter.EmitError<UnknownBaseSpecifier>(int_part.begin());
    return false;
  }
  return true;
}

// Check the integer part (before the '.', if any) is valid.
auto LexedNumericLiteral::Parser::CheckIntPart() -> bool {
  auto int_result = CheckDigitSequence(int_part, radix);
  mantissa_needs_cleaning |= int_result.has_digit_separators;
  return int_result.ok;
}

// Check the fractional part (after the '.' and before the exponent, if any)
// is valid.
auto LexedNumericLiteral::Parser::CheckFractionalPart() -> bool {
  if (IsInteger()) {
    return true;
  }

  if (radix == 2) {
    emitter.EmitError<BinaryRealLiteral>(literal.text.begin() +
                                         literal.radix_point);
    // Carry on and parse the binary real literal anyway.
  }

  // We need to remove a '.' from the mantissa.
  mantissa_needs_cleaning = true;

  return CheckDigitSequence(fract_part, radix,
                            /*allow_digit_separators=*/false)
      .ok;
}

// Check the exponent part (if any) is valid.
auto LexedNumericLiteral::Parser::CheckExponentPart() -> bool {
  if (literal.exponent == static_cast<int>(literal.text.size())) {
    return true;
  }

  char expected_exponent_kind = (radix == 10 ? 'e' : 'p');
  if (literal.text[literal.exponent] != expected_exponent_kind) {
    emitter.EmitError<WrongRealLiteralExponent>(
        literal.text.begin() + literal.exponent,
        {.expected = expected_exponent_kind});
    return false;
  }

  auto exponent_result = CheckDigitSequence(exponent_part, 10);
  exponent_needs_cleaning = exponent_result.has_digit_separators;
  return exponent_result.ok;
}

// Parse the token and compute its value.
auto LexedNumericLiteral::ComputeValue(
    DiagnosticEmitter<const char*>& emitter) const -> Value {
  Parser parser(emitter, *this);

  if (not parser.Check()) {
    return UnrecoverableError();
  }

  if (parser.IsInteger()) {
    return IntegerValue{.value = parser.GetMantissa()};
  }

  return RealValue{.radix = (parser.GetRadix() == 10 ? 10 : 2),
                   .mantissa = parser.GetMantissa(),
                   .exponent = parser.GetExponent()};
}

}  // namespace Carbon
