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
  return arg.name == std::string(name) && isa<AutoPattern>(*arg.pattern);
}

static auto FakeSourceLoc(int line_num) -> SourceLocation {
  return SourceLocation("<test>", line_num);
}

class PatternTest : public ::testing::Test {
 protected:
  Arena arena;
};

TEST_F(PatternTest, EmptyAsPattern) {
  ParenContents<Pattern> contents = {.elements = {},
                                     .has_trailing_comma = false};
  Ptr<const Pattern> pattern =
      PatternFromParenContents(&arena, FakeSourceLoc(1), contents);
  EXPECT_EQ(pattern->SourceLoc(), FakeSourceLoc(1));
  ASSERT_TRUE(isa<TuplePattern>(*pattern));
  EXPECT_THAT(cast<TuplePattern>(*pattern).Fields(), IsEmpty());
}

TEST_F(PatternTest, EmptyAsTuplePattern) {
  ParenContents<Pattern> contents = {.elements = {},
                                     .has_trailing_comma = false};
  Ptr<const TuplePattern> tuple =
      TuplePatternFromParenContents(&arena, FakeSourceLoc(1), contents);
  EXPECT_EQ(tuple->SourceLoc(), FakeSourceLoc(1));
  EXPECT_THAT(tuple->Fields(), IsEmpty());
}

TEST_F(PatternTest, UnaryNoCommaAsPattern) {
  // Equivalent to a code fragment like
  // ```
  // (
  //   auto
  // )
  // ```
  ParenContents<Pattern> contents = {
      .elements = {{.name = std::nullopt,
                    .term = arena.New<AutoPattern>(FakeSourceLoc(2))}},
      .has_trailing_comma = false};

  Ptr<const Pattern> pattern =
      PatternFromParenContents(&arena, FakeSourceLoc(1), contents);
  EXPECT_EQ(pattern->SourceLoc(), FakeSourceLoc(2));
  ASSERT_TRUE(isa<AutoPattern>(*pattern));
}

TEST_F(PatternTest, UnaryNoCommaAsTuplePattern) {
  ParenContents<Pattern> contents = {
      .elements = {{.name = std::nullopt,
                    .term = arena.New<AutoPattern>(FakeSourceLoc(2))}},
      .has_trailing_comma = false};

  Ptr<const TuplePattern> tuple =
      TuplePatternFromParenContents(&arena, FakeSourceLoc(1), contents);
  EXPECT_EQ(tuple->SourceLoc(), FakeSourceLoc(1));
  EXPECT_THAT(tuple->Fields(), ElementsAre(AutoFieldNamed("0")));
}

TEST_F(PatternTest, UnaryWithCommaAsPattern) {
  ParenContents<Pattern> contents = {
      .elements = {{.name = std::nullopt,
                    .term = arena.New<AutoPattern>(FakeSourceLoc(2))}},
      .has_trailing_comma = true};

  Ptr<const Pattern> pattern =
      PatternFromParenContents(&arena, FakeSourceLoc(1), contents);
  EXPECT_EQ(pattern->SourceLoc(), FakeSourceLoc(1));
  ASSERT_TRUE(isa<TuplePattern>(*pattern));
  EXPECT_THAT(cast<TuplePattern>(*pattern).Fields(),
              ElementsAre(AutoFieldNamed("0")));
}

TEST_F(PatternTest, UnaryWithCommaAsTuplePattern) {
  ParenContents<Pattern> contents = {
      .elements = {{.name = std::nullopt,
                    .term = arena.New<AutoPattern>(FakeSourceLoc(2))}},
      .has_trailing_comma = true};

  Ptr<const TuplePattern> tuple =
      TuplePatternFromParenContents(&arena, FakeSourceLoc(1), contents);
  EXPECT_EQ(tuple->SourceLoc(), FakeSourceLoc(1));
  EXPECT_THAT(tuple->Fields(), ElementsAre(AutoFieldNamed("0")));
}

TEST_F(PatternTest, BinaryAsPattern) {
  ParenContents<Pattern> contents = {
      .elements = {{.name = std::nullopt,
                    .term = arena.New<AutoPattern>(FakeSourceLoc(2))},
                   {.name = std::nullopt,
                    .term = arena.New<AutoPattern>(FakeSourceLoc(2))}},
      .has_trailing_comma = true};

  Ptr<const Pattern> pattern =
      PatternFromParenContents(&arena, FakeSourceLoc(1), contents);
  EXPECT_EQ(pattern->SourceLoc(), FakeSourceLoc(1));
  ASSERT_TRUE(isa<TuplePattern>(*pattern));
  EXPECT_THAT(cast<TuplePattern>(*pattern).Fields(),
              ElementsAre(AutoFieldNamed("0"), AutoFieldNamed("1")));
}

TEST_F(PatternTest, BinaryAsTuplePattern) {
  ParenContents<Pattern> contents = {
      .elements = {{.name = std::nullopt,
                    .term = arena.New<AutoPattern>(FakeSourceLoc(2))},
                   {.name = std::nullopt,
                    .term = arena.New<AutoPattern>(FakeSourceLoc(2))}},
      .has_trailing_comma = true};

  Ptr<const TuplePattern> tuple =
      TuplePatternFromParenContents(&arena, FakeSourceLoc(1), contents);
  EXPECT_EQ(tuple->SourceLoc(), FakeSourceLoc(1));
  EXPECT_THAT(tuple->Fields(),
              ElementsAre(AutoFieldNamed("0"), AutoFieldNamed("1")));
}

}  // namespace
}  // namespace Carbon
