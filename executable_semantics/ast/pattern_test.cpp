// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/pattern.h"

#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/paren_contents.h"
#include "executable_semantics/common/arena.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/Support/Casting.h"

namespace Carbon {
namespace {

using llvm::cast;
using llvm::isa;
using testing::ElementsAre;
using testing::IsEmpty;

// Matches a TuplePattern::Field named `name` whose `pattern` is an
// `AutoPattern`.
MATCHER_P(AutoFieldNamed, name, "") {
  return arg.name == std::string(name) && isa<AutoPattern>(arg.pattern);
}

TEST(PatternTest, EmptyAsPattern) {
  ParenContents<Pattern> contents = {.elements = {},
                                     .has_trailing_comma = false};
  const Pattern* pattern = PatternFromParenContents(/*line_num=*/1, contents);
  EXPECT_EQ(pattern->LineNumber(), 1);
  ASSERT_TRUE(isa<TuplePattern>(pattern));
  EXPECT_THAT(cast<TuplePattern>(pattern)->Fields(), IsEmpty());
}

TEST(PatternTest, EmptyAsTuplePattern) {
  ParenContents<Pattern> contents = {.elements = {},
                                     .has_trailing_comma = false};
  const TuplePattern* tuple =
      TuplePatternFromParenContents(/*line_num=*/1, contents);
  EXPECT_EQ(tuple->LineNumber(), 1);
  EXPECT_THAT(tuple->Fields(), IsEmpty());
}

TEST(PatternTest, UnaryNoCommaAsPattern) {
  // Equivalent to a code fragment like
  // ```
  // (
  //   auto
  // )
  // ```
  ParenContents<Pattern> contents = {
      .elements = {{.name = std::nullopt,
                    .term = global_arena->RawNew<AutoPattern>(/*line_num=*/2)}},
      .has_trailing_comma = false};

  const Pattern* pattern = PatternFromParenContents(/*line_num=*/1, contents);
  EXPECT_EQ(pattern->LineNumber(), 2);
  ASSERT_TRUE(isa<AutoPattern>(pattern));
}

TEST(PatternTest, UnaryNoCommaAsTuplePattern) {
  ParenContents<Pattern> contents = {
      .elements = {{.name = std::nullopt,
                    .term = global_arena->RawNew<AutoPattern>(/*line_num=*/2)}},
      .has_trailing_comma = false};

  const TuplePattern* tuple =
      TuplePatternFromParenContents(/*line_num=*/1, contents);
  EXPECT_EQ(tuple->LineNumber(), 1);
  EXPECT_THAT(tuple->Fields(), ElementsAre(AutoFieldNamed("0")));
}

TEST(PatternTest, UnaryWithCommaAsPattern) {
  ParenContents<Pattern> contents = {
      .elements = {{.name = std::nullopt,
                    .term = global_arena->RawNew<AutoPattern>(/*line_num=*/2)}},
      .has_trailing_comma = true};

  const Pattern* pattern = PatternFromParenContents(/*line_num=*/1, contents);
  EXPECT_EQ(pattern->LineNumber(), 1);
  ASSERT_TRUE(isa<TuplePattern>(pattern));
  EXPECT_THAT(cast<TuplePattern>(pattern)->Fields(),
              ElementsAre(AutoFieldNamed("0")));
}

TEST(PatternTest, UnaryWithCommaAsTuplePattern) {
  ParenContents<Pattern> contents = {
      .elements = {{.name = std::nullopt,
                    .term = global_arena->RawNew<AutoPattern>(/*line_num=*/2)}},
      .has_trailing_comma = true};

  const TuplePattern* tuple =
      TuplePatternFromParenContents(/*line_num=*/1, contents);
  EXPECT_EQ(tuple->LineNumber(), 1);
  EXPECT_THAT(tuple->Fields(), ElementsAre(AutoFieldNamed("0")));
}

TEST(PatternTest, BinaryAsPattern) {
  ParenContents<Pattern> contents = {
      .elements = {{.name = std::nullopt,
                    .term = global_arena->RawNew<AutoPattern>(/*line_num=*/2)},
                   {.name = std::nullopt,
                    .term = global_arena->RawNew<AutoPattern>(/*line_num=*/3)}},
      .has_trailing_comma = true};

  const Pattern* pattern = PatternFromParenContents(/*line_num=*/1, contents);
  EXPECT_EQ(pattern->LineNumber(), 1);
  ASSERT_TRUE(isa<TuplePattern>(pattern));
  EXPECT_THAT(cast<TuplePattern>(pattern)->Fields(),
              ElementsAre(AutoFieldNamed("0"), AutoFieldNamed("1")));
}

TEST(PatternTest, BinaryAsTuplePattern) {
  ParenContents<Pattern> contents = {
      .elements = {{.name = std::nullopt,
                    .term = global_arena->RawNew<AutoPattern>(/*line_num=*/2)},
                   {.name = std::nullopt,
                    .term = global_arena->RawNew<AutoPattern>(/*line_num=*/3)}},
      .has_trailing_comma = true};

  const TuplePattern* tuple =
      TuplePatternFromParenContents(/*line_num=*/1, contents);
  EXPECT_EQ(tuple->LineNumber(), 1);
  EXPECT_THAT(tuple->Fields(),
              ElementsAre(AutoFieldNamed("0"), AutoFieldNamed("1")));
}

}  // namespace
}  // namespace Carbon
