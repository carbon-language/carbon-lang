// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TESTING_BASE_UNIFIED_DIFF_MATCHER_H_
#define CARBON_TESTING_BASE_UNIFIED_DIFF_MATCHER_H_

#include <gmock/gmock.h>

#include <algorithm>
#include <optional>
#include <string>
#include <utility>

#include "common/check.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

namespace Carbon::Testing {

// Matcher that compares the elements of two containers and produces a unified
// diff on failure.
template <typename Container>
class UnifiedDiffMatcher {
 public:
  explicit UnifiedDiffMatcher(Container expected)
      : expected_(std::move(expected)) {}

  // Matches `actual` against `expected_`. Returns true on a match; returns
  // false and prints a unified diff to `listener` on a mismatch.
  template <typename ActualContainer>
  auto MatchAndExplain(const ActualContainer& actual,
                       testing::MatchResultListener* listener) const -> bool;

  auto DescribeTo(std::ostream* os) const -> void {
    *os << "matches elements with unified diff";
  }

  auto DescribeNegationTo(std::ostream* os) const -> void {
    *os << "does not match elements with unified diff";
  }

 private:
  // A 2D array, stored contiguously. Rows correspond to `expected_`'s elements,
  // and columns correspond to the actual container's elements.
  template <typename T>
  class Table;

  // The result of a `Matches` check between an expected and actual element.
  enum class MatchResult : uint8_t { Unknown, Matches, DoesNotMatch };

  // Checks whether `actual_element` matches `expected_[expected_index]`. It
  // first checks whether a cached result exists. If not, it evaluates the
  // match and stores the result in `match_results`.
  template <typename ActualElement>
  auto IsElementMatch(size_t expected_index, size_t actual_index,
                      const ActualElement& actual_element,
                      Table<MatchResult>& match_results) const -> bool {
    MatchResult cached_result = match_results.Get(expected_index, actual_index);
    if (cached_result != MatchResult::Unknown) {
      return cached_result == MatchResult::Matches;
    }
    bool is_match =
        testing::MatcherCast<const ActualElement&>(expected_[expected_index])
            .Matches(actual_element);
    match_results.Set(
        expected_index, actual_index,
        is_match ? MatchResult::Matches : MatchResult::DoesNotMatch);
    return is_match;
  }

  // Returns true if every element in `expected_` matches the corresponding
  // element in `actual`. Stores comparisons in `match_results`.
  template <typename ActualContainer>
  auto IsEqual(const ActualContainer& actual,
               Table<MatchResult>& match_results) const -> bool;

  // Populates `subsequences` with the longest common matching subsequences
  // found when comparing `actual` and `expected_`. Stores comparisons in
  // `match_results`.
  template <typename ActualContainer>
  auto GetLongestCommonSubsequences(const ActualContainer& actual,
                                    Table<MatchResult>& match_results,
                                    Table<int>& subsequences) const -> void;

  // Prints the unified diff.
  template <typename ActualContainer>
  auto PrintDiff(const ActualContainer& actual,
                 Table<MatchResult>& match_results,
                 const Table<int>& subsequences,
                 testing::MatchResultListener* listener) const -> void;

  // The expected elements.
  Container expected_;
};

// Returns a polymorphic matcher that acts similarly to
// ElementsAreArray but produces a unified diff on failure.
template <typename Container>
auto ElementsAreArrayWithUnifiedDiff(Container expected) {
  return testing::MakePolymorphicMatcher(
      UnifiedDiffMatcher<Container>(std::move(expected)));
}

// -----------------------------------------------------------------------------
// Internal implementation details follow.
// -----------------------------------------------------------------------------

template <typename Container>
template <typename T>
class UnifiedDiffMatcher<Container>::Table {
 public:
  // Constructs a table with dimensions of expected_size and actual_size,
  // corresponding to the containers being compared.
  Table(int expected_size, int actual_size, T default_value)
      : actual_size_(actual_size),
        data_(expected_size * actual_size, default_value) {}

  // Sets the value at the given expected_index and actual_index.
  auto Set(int expected_index, int actual_index, T value) -> void {
    data_[expected_index * actual_size_ + actual_index] = std::move(value);
  }

  // Gets the value at the given expected_index and actual_index.
  auto Get(int expected_index, int actual_index) const -> T {
    return data_[expected_index * actual_size_ + actual_index];
  }

 private:
  // The actual_size of the table.
  int actual_size_;
  // The contiguous data storage for the table.
  llvm::SmallVector<T> data_;
};

template <typename Container>
template <typename ActualContainer>
auto UnifiedDiffMatcher<Container>::MatchAndExplain(
    const ActualContainer& actual, testing::MatchResultListener* listener) const
    -> bool {
  Table<MatchResult> match_results(expected_.size(), std::size(actual),
                                   MatchResult::Unknown);

  if (IsEqual(actual, match_results)) {
    return true;
  }

  if (listener->IsInterested()) {
    Table<int> subsequences(expected_.size() + 1, std::size(actual) + 1, 0);
    GetLongestCommonSubsequences(actual, match_results, subsequences);
    PrintDiff(actual, match_results, subsequences, listener);
  }
  return false;
}

template <typename Container>
template <typename ActualContainer>
auto UnifiedDiffMatcher<Container>::IsEqual(
    const ActualContainer& actual, Table<MatchResult>& match_results) const
    -> bool {
  if (expected_.size() != std::size(actual)) {
    return false;
  }

  for (auto [i, actual_element] : llvm::enumerate(actual)) {
    if (!IsElementMatch(i, i, actual_element, match_results)) {
      return false;
    }
  }
  return true;
}

template <typename Container>
template <typename ActualContainer>
auto UnifiedDiffMatcher<Container>::GetLongestCommonSubsequences(
    const ActualContainer& actual, Table<MatchResult>& match_results,
    Table<int>& subsequences) const -> void {
  for (auto expected_index : llvm::seq(expected_.size())) {
    for (auto [actual_index, actual_element] : llvm::enumerate(actual)) {
      int subsequence_value;
      if (IsElementMatch(expected_index, actual_index, actual_element,
                         match_results)) {
        // If the elements match, the LCS length increases by 1 relative to
        // the prefixes where both elements are excluded.
        subsequence_value = subsequences.Get(expected_index, actual_index) + 1;
      } else {
        // Otherwise, the LCS length is the maximum of the LCS lengths
        // relative to the prefixes where one element is excluded.
        subsequence_value =
            std::max(subsequences.Get(expected_index, actual_index + 1),
                     subsequences.Get(expected_index + 1, actual_index));
      }
      subsequences.Set(expected_index + 1, actual_index + 1, subsequence_value);
    }
  }
}

template <typename Container>
template <typename ActualContainer>
auto UnifiedDiffMatcher<Container>::PrintDiff(
    const ActualContainer& actual, Table<MatchResult>& match_results,
    const Table<int>& subsequences,
    testing::MatchResultListener* listener) const -> void {
  // A line in the diff output.
  struct DiffLine {
    enum class Kind { Match, ActualOnly, ExpectedOnly };
    Kind kind;
    // Only used for `Match` and `ActualOnly`.
    const ActualContainer::value_type* actual_value;
    int expected_index;
  };

  llvm::SmallVector<DiffLine> diff;
  // Reserve a quick upper bound of the size.
  diff.reserve(expected_.size() + std::size(actual));

  // Reconstruct the diff by backtracking from the end of the table.
  int expected_index = expected_.size() - 1;
  int actual_index = std::size(actual) - 1;
  auto actual_it = std::end(actual) - 1;
  while (expected_index >= 0 || actual_index >= 0) {
    auto match_result = (expected_index >= 0 && actual_index >= 0)
                            ? match_results.Get(expected_index, actual_index)
                            : MatchResult::DoesNotMatch;
    CARBON_CHECK(match_result != MatchResult::Unknown);
    if (match_result == MatchResult::Matches) {
      // The element is in both lists for the diff.
      diff.push_back({.kind = DiffLine::Kind::Match,
                      .actual_value = &*actual_it,
                      .expected_index = expected_index});
      --expected_index;
      --actual_index;
      --actual_it;
    } else if (actual_index >= 0 &&
               (expected_index < 0 ||
                subsequences.Get(expected_index + 1, actual_index) >=
                    subsequences.Get(expected_index, actual_index + 1))) {
      // Dropping an element from `actual` preserves the LCS length, so treat it
      // as an insertion.
      diff.push_back({.kind = DiffLine::Kind::ActualOnly,
                      .actual_value = &*actual_it,
                      .expected_index = std::max(0, expected_index)});
      --actual_index;
      --actual_it;
    } else {
      // Otherwise, treat it as a deletion from `expected`.
      diff.push_back({.kind = DiffLine::Kind::ExpectedOnly,
                      .actual_value = nullptr,
                      .expected_index = expected_index});
      --expected_index;
    }
  }

  struct PrintRange {
    int begin;
    int end;
  };
  llvm::SmallVector<PrintRange> print_ranges;

  constexpr int ContextLines = 3;
  for (auto [i, line] :
       llvm::reverse(llvm::zip_equal(llvm::seq<int>(diff.size()), diff))) {
    if (line.kind != DiffLine::Kind::Match) {
      PrintRange range = {
          .begin = std::max(0, i - ContextLines),
          .end = std::min<int>(diff.size() - 1, i + ContextLines)};
      if (print_ranges.empty() || print_ranges.back().begin > range.end + 1) {
        print_ranges.push_back(range);
      } else {
        // Merge diffs with overlapping context.
        print_ranges.back().begin = range.begin;
      }
    }
  }

  *listener << "unified diff (- expected, + actual):\n";
  for (const auto& range : print_ranges) {
    *listener << "=== diff in expected elements "
              << diff[range.end].expected_index + 1 << " to "
              << diff[range.begin].expected_index + 1 << " (1-based index):\n";
    for (auto i : llvm::reverse(llvm::seq_inclusive(range.begin, range.end))) {
      const auto& line = diff[i];
      if (line.kind == DiffLine::Kind::Match) {
        *listener << "  " << *line.actual_value << "\n";
      } else if (line.kind == DiffLine::Kind::ActualOnly) {
        *listener << "+ " << *line.actual_value << "\n";
      } else {
        *listener << "- ";
        expected_[line.expected_index].DescribeTo(listener->stream());
        *listener << "\n";
      }
    }
  }
  *listener << "=== diff end\n";
}

}  // namespace Carbon::Testing

#endif  // CARBON_TESTING_BASE_UNIFIED_DIFF_MATCHER_H_
