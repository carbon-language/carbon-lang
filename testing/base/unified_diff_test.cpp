// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/base/unified_diff.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <sstream>
#include <string>

#include "llvm/ADT/SmallVector.h"

namespace Carbon::Testing {
namespace {

using ::testing::ElementsAreArray;
using ::testing::Matcher;
using ::testing::MatchesRegex;
using ::testing::StrEq;

// Asserts that when expected does not match actual, the string
// representation of the produced diff equals expected_diff.
auto ExpectUnifiedDiff(const llvm::SmallVector<std::string>& actual,
                       const llvm::SmallVector<Matcher<std::string>>& expected,
                       const std::string& expected_diff) -> void {
  std::stringstream ss;
  ss << UnifiedDiff(expected, actual);
  EXPECT_THAT(ss.str(), testing::Eq(expected_diff));
}

TEST(UnifiedDiffTest, Matches) {
  llvm::SmallVector<std::string> actual = {"A", "B", "C"};
  llvm::SmallVector<Matcher<std::string>> expected = {StrEq("A"), StrEq("B"),
                                                      StrEq("C")};
  EXPECT_TRUE(testing::Value(actual, ElementsAreArray(expected)))
      << UnifiedDiff(expected, actual);

  std::stringstream ss;
  ss << UnifiedDiff(expected, actual);
  EXPECT_THAT(ss.str(), testing::Eq(""));
}

TEST(UnifiedDiffTest, MismatchMissing) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 3 (1-based index):
 A
-B
 C
=== diff end
)";
  ExpectUnifiedDiff({"A", "C"}, {StrEq("A"), StrEq("B"), StrEq("C")},
                    ExpectedDiff);
}

TEST(UnifiedDiffTest, MismatchExtra) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 2 (1-based index):
 A
+B
 C
=== diff end
)";
  ExpectUnifiedDiff({"A", "B", "C"}, {StrEq("A"), StrEq("C")}, ExpectedDiff);
}

TEST(UnifiedDiffTest, MismatchBoth) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 2 (1-based index):
 A
-C
+B
=== diff end
)";
  ExpectUnifiedDiff({"A", "B"}, {StrEq("A"), StrEq("C")}, ExpectedDiff);
}

TEST(UnifiedDiffTest, MismatchMultiple) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 5 (1-based index):
 A
-B
+X
 C
-D
+Y
 E
=== diff end
)";
  ExpectUnifiedDiff(
      {"A", "X", "C", "Y", "E"},
      {StrEq("A"), StrEq("B"), StrEq("C"), StrEq("D"), StrEq("E")},
      ExpectedDiff);
}

TEST(UnifiedDiffTest, MismatchLongContext) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 2 to 8 (1-based index):
 1
 2
 3
-X
+4
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

TEST(UnifiedDiffTest, Mismatch5LineContext) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 7 (1-based index):
-X
+0
 1
 2
 3
 4
 5
-Y
+6
=== diff end
)";
  ExpectUnifiedDiff({"0", "1", "2", "3", "4", "5", "6"},
                    {StrEq("X"), StrEq("1"), StrEq("2"), StrEq("3"), StrEq("4"),
                     StrEq("5"), StrEq("Y")},
                    ExpectedDiff);
}

TEST(UnifiedDiffTest, Mismatch6LineContext) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 8 (1-based index):
-X
+0
 1
 2
 3
 4
 5
 6
-Y
+7
=== diff end
)";
  ExpectUnifiedDiff({"0", "1", "2", "3", "4", "5", "6", "7"},
                    {StrEq("X"), StrEq("1"), StrEq("2"), StrEq("3"), StrEq("4"),
                     StrEq("5"), StrEq("6"), StrEq("Y")},
                    ExpectedDiff);
}

TEST(UnifiedDiffTest, Mismatch7LineContext) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 4 (1-based index):
-X
+0
 1
 2
 3
=== diff in expected elements 6 to 9 (1-based index):
 5
 6
 7
-Y
+8
=== diff end
)";
  ExpectUnifiedDiff({"0", "1", "2", "3", "4", "5", "6", "7", "8"},
                    {StrEq("X"), StrEq("1"), StrEq("2"), StrEq("3"), StrEq("4"),
                     StrEq("5"), StrEq("6"), StrEq("7"), StrEq("Y")},
                    ExpectedDiff);
}

TEST(UnifiedDiffTest, MismatchEmptyExpected) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 1 (1-based index):
+A
=== diff end
)";
  ExpectUnifiedDiff({"A"}, {}, ExpectedDiff);
}

TEST(UnifiedDiffTest, MismatchEmptyActual) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 1 (1-based index):
-A
=== diff end
)";
  ExpectUnifiedDiff({}, {StrEq("A")}, ExpectedDiff);
}

TEST(UnifiedDiffTest, MismatchLongDifference) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 4 (1-based index):
 1
-2
-3
+X
+Y
+Z
 4
=== diff end
)";
  ExpectUnifiedDiff({"1", "X", "Y", "Z", "4"},
                    {StrEq("1"), StrEq("2"), StrEq("3"), StrEq("4")},
                    ExpectedDiff);
}

TEST(UnifiedDiffTest, MismatchGreedyResyncActualMissing) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 6 (1-based index):
 1
 2
-3
+X
+7
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

TEST(UnifiedDiffTest, MismatchGreedyResyncExpectedMissing) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 7 (1-based index):
 1
 2
-X
-7
+3
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

TEST(UnifiedDiffTest, MismatchRegexMatcher) {
  constexpr char ExpectedDiff[] = R"(unified diff (- expected, + actual):
=== diff in expected elements 1 to 3 (1-based index):
 A
-matches regular expression ".*B.*"
 C
=== diff end
)";
  ExpectUnifiedDiff({"A", "C"}, {StrEq("A"), MatchesRegex(".*B.*"), StrEq("C")},
                    ExpectedDiff);
}

TEST(UnifiedDiffTest, CheckSubset) {
  constexpr char ExpectedDiff[] =
      R"(unified diff (- expected, + actual) [+ lines are normal]:
=== diff in expected elements 1 to 4 (1-based index):
-X
+0
 1
 2
 3
=== diff end
)";
  std::stringstream ss;
  llvm::SmallVector<Matcher<std::string>> expected = {
      StrEq("X"), StrEq("1"), StrEq("2"), StrEq("3"), StrEq("4"),
      StrEq("5"), StrEq("6"), StrEq("7"), StrEq("8"), StrEq("9")};
  llvm::SmallVector<std::string> actual = {"0", "1", "2", "3",  "4",  "5", "6",
                                           "7", "8", "9", "10", "11", "12"};
  ss << UnifiedDiff(expected, actual, /*check_subset=*/true);
  EXPECT_THAT(ss.str(), testing::Eq(ExpectedDiff));
}

}  // namespace
}  // namespace Carbon::Testing
