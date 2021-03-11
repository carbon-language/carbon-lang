// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "lexer/numeric_literal.h"

#include <iterator>
#include <memory>
#include <vector>

#include "diagnostics/diagnostic_emitter.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "lexer/test_helpers.h"

namespace Carbon {
namespace {

struct NumericLiteralTest : ::testing::Test {
  std::vector<std::unique_ptr<Testing::SingleTokenDiagnosticTranslator>>
      translators;
  std::vector<std::unique_ptr<DiagnosticEmitter<const char*>>> emitters;

  auto Lex(llvm::StringRef text) -> NumericLiteralToken {
    llvm::Optional<NumericLiteralToken> result = NumericLiteralToken::Lex(text);
    assert(result);
    EXPECT_EQ(result->Text(), text);
    return *result;
  }

  auto Parse(llvm::StringRef text) -> NumericLiteralToken::Parser {
    translators.push_back(
        std::make_unique<Testing::SingleTokenDiagnosticTranslator>(text));
    emitters.push_back(std::make_unique<DiagnosticEmitter<const char*>>(
        *translators.back(), ConsoleDiagnosticConsumer()));
    return NumericLiteralToken::Parser(*emitters.back(), Lex(text));
  }
};

TEST_F(NumericLiteralTest, HandlesIntegerLiteral) {
  struct Testcase {
    llvm::StringLiteral token;
    uint64_t value;
    int radix;
  };
  Testcase testcases[] = {
      {.token = "12", .value = 12, .radix = 10},
      {.token = "0x12_3ABC", .value = 0x12'3ABC, .radix = 16},
      {.token = "0b10_10_11", .value = 0b10'10'11, .radix = 2},
      {.token = "1_234_567", .value = 1'234'567, .radix = 10},
  };
  for (Testcase testcase : testcases) {
    auto parser = Parse(testcase.token);
    EXPECT_EQ(parser.Check(), parser.Valid) << testcase.token;
    EXPECT_EQ(parser.IsInteger(), true);
    EXPECT_EQ(parser.GetMantissa().getZExtValue(), testcase.value);
    EXPECT_EQ(parser.GetExponent().getSExtValue(), 0);
    EXPECT_EQ(parser.GetRadix(), testcase.radix);
  }
}

TEST_F(NumericLiteralTest, ValidatesBaseSpecifier) {
  llvm::StringLiteral valid[] = {
      // Decimal integer literals.
      "0",
      "1",
      "123456789000000000000000000000000000000000000",

      // Hexadecimal integer literals.
      "0x0123456789ABCDEF",
      "0x0000000000000000000000000000000",

      // Binary integer literals.
      "0b10110100101001010",
      "0b0000000",
  };
  for (llvm::StringLiteral literal : valid) {
    auto parser = Parse(literal);
    EXPECT_EQ(parser.Check(), parser.Valid) << literal;
  }

  llvm::StringLiteral invalid[] = {
      "00",  "0X123",    "0o123",          "0B1",
      "007", "123L",     "123456789A",     "0x",
      "0b",  "0x123abc", "0b011101201001", "0b10A",
      "0x_", "0b_",
  };
  for (llvm::StringLiteral literal : invalid) {
    auto parser = Parse(literal);
    EXPECT_EQ(parser.Check(), parser.UnrecoverableError) << literal;
  }
}

TEST_F(NumericLiteralTest, ValidatesIntegerDigitSeparators) {
  llvm::StringLiteral valid[] = {
      // Decimal literals optionally have digit separators every 3 places.
      "1_234",
      "123_456",
      "1_234_567",

      // Hexadecimal literals optionally have digit separators every 4 places.
      "0x1_0000",
      "0x1000_0000",
      "0x1_0000_0000",

      // Binary integer literals can have digit separators anywhere..
      "0b1_0_1_0_1_0",
      "0b111_0000",
  };
  for (llvm::StringLiteral literal : valid) {
    auto parser = Parse(literal);
    EXPECT_EQ(parser.Check(), parser.Valid) << literal;
  }

  llvm::StringLiteral invalid[] = {
      // Decimal literals.
      "12_34",
      "123_4_6_789",
      "12_3456_789",
      "12__345",
      "1_",

      // Hexadecimal literals.
      "0x_1234",
      "0x123_",
      "0x12_3",
      "0x_234_5678",
      "0x1234_567",

      // Binary literals.
      "0b_10101",
      "0b1__01",
      "0b1011_",
      "0b1_01_01_",
  };
  for (llvm::StringLiteral literal : invalid) {
    auto parser = Parse(literal);
    EXPECT_EQ(parser.Check(), parser.RecoverableError) << literal;
  }
}

TEST_F(NumericLiteralTest, HandlesRealLiteral) {
  struct Testcase {
    llvm::StringLiteral token;
    uint64_t mantissa;
    int64_t exponent;
    unsigned radix;
  };
  Testcase testcases[] = {
      // Decimal real literals.
      {.token = "0.0", .mantissa = 0, .exponent = -1, .radix = 10},
      {.token = "12.345", .mantissa = 12345, .exponent = -3, .radix = 10},
      {.token = "12.345e6", .mantissa = 12345, .exponent = 3, .radix = 10},
      {.token = "12.345e+6", .mantissa = 12345, .exponent = 3, .radix = 10},
      {.token = "1_234.5e-2", .mantissa = 12345, .exponent = -3, .radix = 10},
      {.token = "1.0e-2_000_000",
       .mantissa = 10,
       .exponent = -2'000'001,
       .radix = 10},

      // Hexadecimal real literals.
      {.token = "0x1_2345_6789.CDEF",
       .mantissa = 0x1'2345'6789'CDEF,
       .exponent = -16,
       .radix = 16},
      {.token = "0x0.0001p4", .mantissa = 1, .exponent = -12, .radix = 16},
      {.token = "0x0.0001p+4", .mantissa = 1, .exponent = -12, .radix = 16},
      {.token = "0x0.0001p-4", .mantissa = 1, .exponent = -20, .radix = 16},
      // The exponent here works out as exactly INT64_MIN.
      {.token = "0x1.01p-9223372036854775800",
       .mantissa = 0x101,
       .exponent = -9223372036854775807L - 1L,
       .radix = 16},
      // The exponent here doesn't fit in a signed 64-bit integer until we
      // adjust for the radix point.
      {.token = "0x1.01p9223372036854775809",
       .mantissa = 0x101,
       .exponent = 9223372036854775801L,
       .radix = 16},

      // Binary real literals. These are invalid, but we accept them for error
      // recovery.
      {.token = "0b10_11_01.01",
       .mantissa = 0b10110101,
       .exponent = -2,
       .radix = 2},
  };
  for (Testcase testcase : testcases) {
    auto parser = Parse(testcase.token);
    EXPECT_EQ(parser.Check(),
              testcase.radix == 2 ? parser.RecoverableError : parser.Valid)
        << testcase.token;
    EXPECT_EQ(parser.IsInteger(), false);
    EXPECT_EQ(parser.GetMantissa().getZExtValue(), testcase.mantissa);
    EXPECT_EQ(parser.GetExponent().getSExtValue(), testcase.exponent);
    EXPECT_EQ(parser.GetRadix(), testcase.radix);
  }
}

TEST_F(NumericLiteralTest, HandlesRealLiteralOverflow) {
  llvm::StringLiteral input = "0x1.000001p-9223372036854775800";
  auto parser = Parse(input);
  EXPECT_EQ(parser.Check(), parser.Valid);
  EXPECT_EQ(parser.GetMantissa(), 0x1000001);
  EXPECT_EQ((parser.GetExponent() + 9223372036854775800).getSExtValue(), -24);
  EXPECT_EQ(parser.GetRadix(), 16);
}

TEST_F(NumericLiteralTest, ValidatesRealLiterals) {
  llvm::StringLiteral invalid_digit_separators[] = {
      // Invalid digit separators.
      "12_34.5",     "123.4_567", "123.456_7", "1_2_3.4",
      "123.4e56_78", "0x12_34.5", "0x12.3_4",  "0x12.34p5_6",
  };
  for (llvm::StringLiteral literal : invalid_digit_separators) {
    auto parser = Parse(literal);
    EXPECT_EQ(parser.Check(), parser.RecoverableError) << literal;
  }

  llvm::StringLiteral invalid[] = {
      // No digits in integer part.
      "0x.0",
      "0b.0",
      "0x_.0",
      "0b_.0",

      // No digits in fractional part.
      "0.e",
      "0.e0",
      "0.e+0",
      "0x0.p",
      "0x0.p-0",

      // Invalid digits in mantissa.
      "123A.4",
      "123.4A",
      "123A.4e0",
      "123.4Ae0",
      "0x123ABCDEFG.0",
      "0x123.ABCDEFG",
      "0x123ABCDEFG.0p0",
      "0x123.ABCDEFGp0",

      // Invalid exponent letter.
      "0.0f0",
      "0.0p0",
      "0.0z+0",
      "0x0.0e0",
      "0x0.0f0",
      "0x0.0z-0",

      // No digits in exponent part.
      "0.0e",
      "0x0.0p",
      "0.0e_",
      "0x0.0p_",

      // Invalid digits in exponent part.
      "0.0eHELLO",
      "0.0eA",
      "0.0e+A",
      "0x0.0pA",
      "0x0.0p-A",
  };
  for (llvm::StringLiteral literal : invalid) {
    auto parser = Parse(literal);
    EXPECT_EQ(parser.Check(), parser.UnrecoverableError) << literal;
  }
}

}  // namespace
}  // namespace Carbon
