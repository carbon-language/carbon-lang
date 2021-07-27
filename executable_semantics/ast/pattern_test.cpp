// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/pattern.h"

#include "executable_semantics/ast/expression.h"
#include "executable_semantics/syntax/paren_contents.h"
#include "gtest/gtest.h"
#include "llvm/Support/Casting.h"

namespace Carbon {
namespace {

using llvm::cast;
using llvm::isa;

TEST(PatternTest, EmptyAsPattern) {
  ParenContents<Pattern> contents = {.elements = {},
                                     .has_trailing_comma = false};
  const Pattern* pattern = AsPattern(/*line_number=*/1, contents);
  EXPECT_EQ(pattern->LineNumber(), 1);
  ASSERT_TRUE(isa<TuplePattern>(pattern));
  EXPECT_EQ(cast<TuplePattern>(pattern)->Fields().size(), 0);
}

TEST(PatternTest, EmptyAsTuplePattern) {
  ParenContents<Pattern> contents = {.elements = {},
                                     .has_trailing_comma = false};
  const TuplePattern* tuple = AsTuplePattern(/*line_number=*/1, contents);
  EXPECT_EQ(tuple->LineNumber(), 1);
  EXPECT_EQ(tuple->Fields().size(), 0);
}

TEST(PatternTest, UnaryNoCommaAsPattern) {
  // Equivalent to a code fragment like
  // ```
  // (
  //   42
  // )
  // ```
  ParenContents<Pattern> contents = {
      .elements = {{.name = std::nullopt,
                    .term = new AutoPattern(/*line_number=*/2)}},
      .has_trailing_comma = false};

  const Pattern* pattern = AsPattern(/*line_number=*/1, contents);
  EXPECT_EQ(pattern->LineNumber(), 2);
  ASSERT_TRUE(isa<AutoPattern>(pattern));
}

TEST(PatternTest, UnaryNoCommaAsTuplePattern) {
  ParenContents<Pattern> contents = {
      .elements = {{.name = std::nullopt,
                    .term = new AutoPattern(/*line_number=*/2)}},
      .has_trailing_comma = false};

  const TuplePattern* tuple = AsTuplePattern(/*line_number=*/1, contents);
  EXPECT_EQ(tuple->LineNumber(), 1);
  const std::vector<TuplePattern::Field>& fields = tuple->Fields();
  ASSERT_EQ(fields.size(), 1);
  EXPECT_EQ(fields[0].name, "0");
  EXPECT_TRUE(isa<AutoPattern>(fields[0].pattern));
}

TEST(PatternTest, UnaryWithCommaAsPattern) {
  ParenContents<Pattern> contents = {
      .elements = {{.name = std::nullopt,
                    .term = new AutoPattern(/*line_number=*/2)}},
      .has_trailing_comma = true};

  const Pattern* pattern = AsPattern(/*line_number=*/1, contents);
  EXPECT_EQ(pattern->LineNumber(), 1);
  ASSERT_TRUE(isa<TuplePattern>(pattern));
  const std::vector<TuplePattern::Field>& fields =
      cast<TuplePattern>(pattern)->Fields();
  ASSERT_EQ(fields.size(), 1);
  EXPECT_EQ(fields[0].name, "0");
  EXPECT_TRUE(isa<AutoPattern>(fields[0].pattern));
}

TEST(PatternTest, UnaryWithCommaAsTuplePattern) {
  ParenContents<Pattern> contents = {
      .elements = {{.name = std::nullopt,
                    .term = new AutoPattern(/*line_number=*/2)}},
      .has_trailing_comma = true};

  const TuplePattern* tuple = AsTuplePattern(/*line_number=*/1, contents);
  EXPECT_EQ(tuple->LineNumber(), 1);
  const std::vector<TuplePattern::Field>& fields = tuple->Fields();
  EXPECT_EQ(fields[0].name, "0");
  EXPECT_TRUE(isa<AutoPattern>(fields[0].pattern));
}

TEST(PatternTest, BinaryAsPattern) {
  ParenContents<Pattern> contents = {
      .elements = {{.name = std::nullopt,
                    .term = new AutoPattern(/*line_number=*/2)},
                   {.name = std::nullopt,
                    .term = new AutoPattern(/*line_number=*/3)}},
      .has_trailing_comma = true};

  const Pattern* pattern = AsPattern(/*line_number=*/1, contents);
  EXPECT_EQ(pattern->LineNumber(), 1);
  ASSERT_TRUE(isa<TuplePattern>(pattern));
  const std::vector<TuplePattern::Field>& fields =
      cast<TuplePattern>(pattern)->Fields();
  ASSERT_EQ(fields.size(), 2);
  EXPECT_EQ(fields[0].name, "0");
  EXPECT_TRUE(isa<AutoPattern>(fields[0].pattern));
  EXPECT_EQ(fields[1].name, "1");
  EXPECT_TRUE(isa<AutoPattern>(fields[1].pattern));
}

TEST(PatternTest, BinaryAsTuplePattern) {
  ParenContents<Pattern> contents = {
      .elements = {{.name = std::nullopt,
                    .term = new AutoPattern(/*line_number=*/2)},
                   {.name = std::nullopt,
                    .term = new AutoPattern(/*line_number=*/3)}},
      .has_trailing_comma = true};

  const TuplePattern* tuple = AsTuplePattern(/*line_number=*/1, contents);
  EXPECT_EQ(tuple->LineNumber(), 1);
  const std::vector<TuplePattern::Field>& fields = tuple->Fields();
  ASSERT_EQ(fields.size(), 2);
  EXPECT_EQ(fields[0].name, "0");
  EXPECT_TRUE(isa<AutoPattern>(fields[0].pattern));
  EXPECT_EQ(fields[1].name, "1");
  EXPECT_TRUE(isa<AutoPattern>(fields[1].pattern));
}

}  // namespace
}  // namespace Carbon
