// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/base/union_diff_matcher.h"

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
auto ExpectUnionDiff(const llvm::SmallVector<std::string>& actual,
                     const llvm::SmallVector<Matcher<std::string>>& expected,
                     const std::string& expected_diff) -> void {
  testing::StringMatchResultListener listener;
  EXPECT_FALSE(testing::ExplainMatchResult(
      ElementsAreArrayWithUnionDiff(expected), actual, &listener));
  EXPECT_THAT(listener.str(), testing::Eq(expected_diff));
}

TEST(UnionDiffMatcherTest, Matches) {
  llvm::SmallVector<std::string> actual = {"A", "B", "C"};
  llvm::SmallVector<Matcher<std::string>> expected = {StrEq("A"), StrEq("B"),
                                                      StrEq("C")};
  EXPECT_THAT(actual, ElementsAreArrayWithUnionDiff(expected));
}

TEST(UnionDiffMatcherTest, MismatchMissing) {
  constexpr char ExpectedDiff[] = R"(union diff (- expected, + actual):
=== diff in expected elements 1 to 3 (1-based index):
  A
- is equal to "B"
  C
=== diff end
)";
  ExpectUnionDiff({"A", "C"}, {StrEq("A"), StrEq("B"), StrEq("C")},
                  ExpectedDiff);
}

TEST(UnionDiffMatcherTest, MismatchExtra) {
  constexpr char ExpectedDiff[] = R"(union diff (- expected, + actual):
=== diff in expected elements 1 to 2 (1-based index):
  A
+ B
  C
=== diff end
)";
  ExpectUnionDiff({"A", "B", "C"}, {StrEq("A"), StrEq("C")}, ExpectedDiff);
}

TEST(UnionDiffMatcherTest, MismatchBoth) {
  constexpr char ExpectedDiff[] = R"(union diff (- expected, + actual):
=== diff in expected elements 1 to 2 (1-based index):
  A
- is equal to "C"
+ B
=== diff end
)";
  ExpectUnionDiff({"A", "B"}, {StrEq("A"), StrEq("C")}, ExpectedDiff);
}

TEST(UnionDiffMatcherTest, MismatchMultiple) {
  constexpr char ExpectedDiff[] = R"(union diff (- expected, + actual):
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
  ExpectUnionDiff({"A", "X", "C", "Y", "E"},
                  {StrEq("A"), StrEq("B"), StrEq("C"), StrEq("D"), StrEq("E")},
                  ExpectedDiff);
}

TEST(UnionDiffMatcherTest, MismatchLongContext) {
  constexpr char ExpectedDiff[] = R"(union diff (- expected, + actual):
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
  ExpectUnionDiff({"0", "1", "2", "3", "4", "5", "6", "7", "8"},
                  {StrEq("0"), StrEq("1"), StrEq("2"), StrEq("3"), StrEq("X"),
                   StrEq("5"), StrEq("6"), StrEq("7"), StrEq("8")},
                  ExpectedDiff);
}

TEST(UnionDiffMatcherTest, Mismatch5LineContext) {
  constexpr char ExpectedDiff[] = R"(union diff (- expected, + actual):
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
  ExpectUnionDiff({"0", "1", "2", "3", "4", "5", "6"},
                  {StrEq("X"), StrEq("1"), StrEq("2"), StrEq("3"), StrEq("4"),
                   StrEq("5"), StrEq("Y")},
                  ExpectedDiff);
}

TEST(UnionDiffMatcherTest, Mismatch6LineContext) {
  constexpr char ExpectedDiff[] = R"(union diff (- expected, + actual):
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
  ExpectUnionDiff({"0", "1", "2", "3", "4", "5", "6", "7"},
                  {StrEq("X"), StrEq("1"), StrEq("2"), StrEq("3"), StrEq("4"),
                   StrEq("5"), StrEq("6"), StrEq("Y")},
                  ExpectedDiff);
}

TEST(UnionDiffMatcherTest, Mismatch7LineContext) {
  constexpr char ExpectedDiff[] = R"(union diff (- expected, + actual):
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
  ExpectUnionDiff({"0", "1", "2", "3", "4", "5", "6", "7", "8"},
                  {StrEq("X"), StrEq("1"), StrEq("2"), StrEq("3"), StrEq("4"),
                   StrEq("5"), StrEq("6"), StrEq("7"), StrEq("Y")},
                  ExpectedDiff);
}

TEST(UnionDiffMatcherTest, MismatchEmptyExpected) {
  constexpr char ExpectedDiff[] = R"(union diff (- expected, + actual):
=== diff in expected elements 1 to 1 (1-based index):
+ A
=== diff end
)";
  ExpectUnionDiff({"A"}, {}, ExpectedDiff);
}

TEST(UnionDiffMatcherTest, MismatchEmptyActual) {
  constexpr char ExpectedDiff[] = R"(union diff (- expected, + actual):
=== diff in expected elements 1 to 1 (1-based index):
- is equal to "A"
=== diff end
)";
  ExpectUnionDiff({}, {StrEq("A")}, ExpectedDiff);
}

TEST(UnionDiffMatcherTest, MismatchLongDifference) {
  constexpr char ExpectedDiff[] = R"(union diff (- expected, + actual):
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
  ExpectUnionDiff({"1", "X", "Y", "Z", "4"},
                  {StrEq("1"), StrEq("2"), StrEq("3"), StrEq("4")},
                  ExpectedDiff);
}

TEST(UnionDiffMatcherTest, MismatchGreedyResyncActualMissing) {
  constexpr char ExpectedDiff[] = R"(union diff (- expected, + actual):
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
  ExpectUnionDiff({"1", "2", "X", "7", "4", "5", "6", "7", "8", "9"},
                  {StrEq("1"), StrEq("2"), StrEq("3"), StrEq("4"), StrEq("5"),
                   StrEq("6"), StrEq("7"), StrEq("8"), StrEq("9")},
                  ExpectedDiff);
}

TEST(UnionDiffMatcherTest, MismatchGreedyResyncExpectedMissing) {
  constexpr char ExpectedDiff[] = R"(union diff (- expected, + actual):
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
  ExpectUnionDiff({"1", "2", "3", "4", "5", "6", "7", "8", "9"},
                  {StrEq("1"), StrEq("2"), StrEq("X"), StrEq("7"), StrEq("4"),
                   StrEq("5"), StrEq("6"), StrEq("7"), StrEq("8"), StrEq("9")},
                  ExpectedDiff);
}

}  // namespace
}  // namespace Carbon::Testing
