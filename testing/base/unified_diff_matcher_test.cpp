// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/base/unified_diff_matcher.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>

#include "llvm/ADT/SmallVector.h"

namespace Carbon::Testing {
namespace {

using ::testing::Matcher;
using ::testing::StrEq;

// Asserts that when expected does not match actual, the string
// representation of the produced diff equals expected_diff.
auto ExpectUnifiedDiff(const llvm::SmallVector<std::string>& actual,
                       const llvm::SmallVector<Matcher<std::string>>& expected,
                       const std::string& expected_diff) -> void {
  testing::StringMatchResultListener listener;
  EXPECT_FALSE(testing::ExplainMatchResult(
      ElementsAreArrayWithUnifiedDiff(expected), actual, &listener));
  EXPECT_THAT(listener.str(), testing::Eq(expected_diff));
}

TEST(UnifiedDiffMatcherTest, Matches) {
  llvm::SmallVector<std::string> actual = {"A", "B", "C"};
  llvm::SmallVector<Matcher<std::string>> expected = {StrEq("A"), StrEq("B"),
                                                      StrEq("C")};
  EXPECT_THAT(actual, ElementsAreArrayWithUnifiedDiff(expected));
}

TEST(UnifiedDiffMatcherTest, MismatchMissing) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 3 (1-based index):
  A
- is equal to "B"
  C
=== diff end
)";
  ExpectUnifiedDiff({"A", "C"}, {StrEq("A"), StrEq("B"), StrEq("C")},
                    ExpectedDiff);
}

TEST(UnifiedDiffMatcherTest, MismatchExtra) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 2 (1-based index):
  A
+ B
  C
=== diff end
)";
  ExpectUnifiedDiff({"A", "B", "C"}, {StrEq("A"), StrEq("C")}, ExpectedDiff);
}

TEST(UnifiedDiffMatcherTest, MismatchBoth) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 2 (1-based index):
  A
- is equal to "C"
+ B
=== diff end
)";
  ExpectUnifiedDiff({"A", "B"}, {StrEq("A"), StrEq("C")}, ExpectedDiff);
}

TEST(UnifiedDiffMatcherTest, MismatchMultiple) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 5 (1-based index):
  A
- is equal to "B"
+ X
  C
- is equal to "D"
+ Y
  E
=== diff end
)";
  ExpectUnifiedDiff(
      {"A", "X", "C", "Y", "E"},
      {StrEq("A"), StrEq("B"), StrEq("C"), StrEq("D"), StrEq("E")},
      ExpectedDiff);
}

TEST(UnifiedDiffMatcherTest, MismatchLongContext) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 2 to 8 (1-based index):
  1
  2
  3
- is equal to "X"
+ 4
  5
  6
  7
=== diff end
)";
  ExpectUnifiedDiff({"0", "1", "2", "3", "4", "5", "6", "7", "8"},
                    {StrEq("0"), StrEq("1"), StrEq("2"), StrEq("3"), StrEq("X"),
                     StrEq("5"), StrEq("6"), StrEq("7"), StrEq("8")},
                    ExpectedDiff);
}

TEST(UnifiedDiffMatcherTest, Mismatch5LineContext) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 7 (1-based index):
- is equal to "X"
+ 0
  1
  2
  3
  4
  5
- is equal to "Y"
+ 6
=== diff end
)";
  ExpectUnifiedDiff({"0", "1", "2", "3", "4", "5", "6"},
                    {StrEq("X"), StrEq("1"), StrEq("2"), StrEq("3"), StrEq("4"),
                     StrEq("5"), StrEq("Y")},
                    ExpectedDiff);
}

TEST(UnifiedDiffMatcherTest, Mismatch6LineContext) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 8 (1-based index):
- is equal to "X"
+ 0
  1
  2
  3
  4
  5
  6
- is equal to "Y"
+ 7
=== diff end
)";
  ExpectUnifiedDiff({"0", "1", "2", "3", "4", "5", "6", "7"},
                    {StrEq("X"), StrEq("1"), StrEq("2"), StrEq("3"), StrEq("4"),
                     StrEq("5"), StrEq("6"), StrEq("Y")},
                    ExpectedDiff);
}

TEST(UnifiedDiffMatcherTest, Mismatch7LineContext) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 4 (1-based index):
- is equal to "X"
+ 0
  1
  2
  3
=== diff in expected elements 6 to 9 (1-based index):
  5
  6
  7
- is equal to "Y"
+ 8
=== diff end
)";
  ExpectUnifiedDiff({"0", "1", "2", "3", "4", "5", "6", "7", "8"},
                    {StrEq("X"), StrEq("1"), StrEq("2"), StrEq("3"), StrEq("4"),
                     StrEq("5"), StrEq("6"), StrEq("7"), StrEq("Y")},
                    ExpectedDiff);
}

TEST(UnifiedDiffMatcherTest, MismatchEmptyExpected) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 1 (1-based index):
+ A
=== diff end
)";
  ExpectUnifiedDiff({"A"}, {}, ExpectedDiff);
}

TEST(UnifiedDiffMatcherTest, MismatchEmptyActual) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 1 (1-based index):
- is equal to "A"
=== diff end
)";
  ExpectUnifiedDiff({}, {StrEq("A")}, ExpectedDiff);
}

TEST(UnifiedDiffMatcherTest, MismatchLongDifference) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 4 (1-based index):
  1
- is equal to "2"
- is equal to "3"
+ X
+ Y
+ Z
  4
=== diff end
)";
  ExpectUnifiedDiff({"1", "X", "Y", "Z", "4"},
                    {StrEq("1"), StrEq("2"), StrEq("3"), StrEq("4")},
                    ExpectedDiff);
}

TEST(UnifiedDiffMatcherTest, MismatchGreedyResyncActualMissing) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 6 (1-based index):
  1
  2
- is equal to "3"
+ X
+ 7
  4
  5
  6
=== diff end
)";
  ExpectUnifiedDiff({"1", "2", "X", "7", "4", "5", "6", "7", "8", "9"},
                    {StrEq("1"), StrEq("2"), StrEq("3"), StrEq("4"), StrEq("5"),
                     StrEq("6"), StrEq("7"), StrEq("8"), StrEq("9")},
                    ExpectedDiff);
}

TEST(UnifiedDiffMatcherTest, MismatchGreedyResyncExpectedMissing) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 7 (1-based index):
  1
  2
- is equal to "X"
- is equal to "7"
+ 3
  4
  5
  6
=== diff end
)";
  ExpectUnifiedDiff(
      {"1", "2", "3", "4", "5", "6", "7", "8", "9"},
      {StrEq("1"), StrEq("2"), StrEq("X"), StrEq("7"), StrEq("4"), StrEq("5"),
       StrEq("6"), StrEq("7"), StrEq("8"), StrEq("9")},
      ExpectedDiff);
}

}  // namespace
}  // namespace Carbon::Testing
