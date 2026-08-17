// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TESTING_BASE_UNIFIED_DIFF_H_
#define CARBON_TESTING_BASE_UNIFIED_DIFF_H_

#include <gmock/gmock.h>

#include <algorithm>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

#include "common/check.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

namespace Carbon::Testing {

// Compares the elements of two containers and prints a unified diff when
// streamed to an ostream.
template <typename ExpectedContainer, typename ActualContainer>
class UnifiedDiff {
 public:
  explicit UnifiedDiff(const ExpectedContainer& expected,
                       const ActualContainer& actual, bool check_subset = false)
      : expected_(expected), actual_(actual), check_subset_(check_subset) {}

  friend auto operator<<(std::ostream& os, const UnifiedDiff& diff)
      -> std::ostream& {
    diff.Print(&os);
    return os;
  }

  // Prints the unified diff to `os`, or prints nothing if `expected_` and
  // `actual_` match.
  auto Print(std::ostream* os) const -> void;

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
  // element in `actual_`. Stores comparisons in `match_results`.
  auto IsEqual(Table<MatchResult>& match_results) const -> bool;

  // Populates `subsequences` with the longest common matching subsequences
  // found when comparing `actual_` and `expected_`. Stores comparisons in
  // `match_results`.
  auto GetLongestCommonSubsequences(Table<MatchResult>& match_results,
                                    Table<int>& subsequences) const -> void;

  // Prints the unified diff.
  auto PrintDiff(Table<MatchResult>& match_results,
                 const Table<int>& subsequences, std::ostream* os) const
      -> void;

  // The expected and actual elements.
  const ExpectedContainer& expected_;
  const ActualContainer& actual_;
  // Whether we are checking that `expected_` is a subset of `actual_`.
  bool check_subset_;
};

template <typename ExpectedContainer, typename ActualContainer>
UnifiedDiff(const ExpectedContainer&, const ActualContainer&)
    -> UnifiedDiff<ExpectedContainer, ActualContainer>;
template <typename ExpectedContainer, typename ActualContainer>
UnifiedDiff(const ExpectedContainer&, const ActualContainer&, bool)
    -> UnifiedDiff<ExpectedContainer, ActualContainer>;

// -----------------------------------------------------------------------------
// Internal implementation details follow.
// -----------------------------------------------------------------------------

template <typename ExpectedContainer, typename ActualContainer>
template <typename T>
class UnifiedDiff<ExpectedContainer, ActualContainer>::Table {
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

template <typename ExpectedContainer, typename ActualContainer>
auto UnifiedDiff<ExpectedContainer, ActualContainer>::Print(
    std::ostream* os) const -> void {
  Table<MatchResult> match_results(expected_.size(), std::size(actual_),
                                   MatchResult::Unknown);

  if (IsEqual(match_results)) {
    return;
  }

  Table<int> subsequences(expected_.size() + 1, std::size(actual_) + 1, 0);
  GetLongestCommonSubsequences(match_results, subsequences);
  PrintDiff(match_results, subsequences, os);
}

template <typename ExpectedContainer, typename ActualContainer>
auto UnifiedDiff<ExpectedContainer, ActualContainer>::IsEqual(
    Table<MatchResult>& match_results) const -> bool {
  if (expected_.size() != std::size(actual_)) {
    return false;
  }

  for (auto [i, actual_element] : llvm::enumerate(actual_)) {
    if (!IsElementMatch(i, i, actual_element, match_results)) {
      return false;
    }
  }
  return true;
}

template <typename ExpectedContainer, typename ActualContainer>
auto UnifiedDiff<ExpectedContainer, ActualContainer>::
    GetLongestCommonSubsequences(Table<MatchResult>& match_results,
                                 Table<int>& subsequences) const -> void {
  for (auto expected_index : llvm::seq(expected_.size())) {
    for (auto [actual_index, actual_element] : llvm::enumerate(actual_)) {
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

template <typename ExpectedContainer, typename ActualContainer>
auto UnifiedDiff<ExpectedContainer, ActualContainer>::PrintDiff(
    Table<MatchResult>& match_results, const Table<int>& subsequences,
    std::ostream* os) const -> void {
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
  diff.reserve(expected_.size() + std::size(actual_));

  // Reconstruct the diff by backtracking from the end of the table.
  int expected_index = expected_.size() - 1;
  int actual_index = std::size(actual_) - 1;
  auto actual_it = std::end(actual_) - 1;
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

  *os << "unified diff (- expected, + actual)";
  if (check_subset_) {
    *os << " [+ lines are normal]";
  }
  *os << ":\n";
  for (const auto& range : print_ranges) {
    if (check_subset_) {
      // In check_subset mode, only print diff ranges that contain unmatched
      // expected lines.
      bool has_expected_only = false;
      for (auto i : llvm::seq_inclusive(range.begin, range.end)) {
        if (diff[i].kind == DiffLine::Kind::ExpectedOnly) {
          has_expected_only = true;
          break;
        }
      }
      if (!has_expected_only) {
        continue;
      }
    }
    *os << "=== diff in expected elements "
        << diff[range.end].expected_index + 1 << " to "
        << diff[range.begin].expected_index + 1 << " (1-based index):\n";
    for (auto i : llvm::reverse(llvm::seq_inclusive(range.begin, range.end))) {
      const auto& line = diff[i];
      if (line.kind == DiffLine::Kind::Match) {
        *os << " " << *line.actual_value << "\n";
      } else if (line.kind == DiffLine::Kind::ActualOnly) {
        *os << "+" << *line.actual_value << "\n";
      } else {
        *os << "-";
        // Strip off the extra decoration that `StrEq` adds.
        // TODO: Also tidy up the `MatchesRegex` description. Maybe we shouldn't
        // be building a list of matchers at all.
        std::stringstream ss;
        expected_[line.expected_index].DescribeTo(&ss);
        std::string desc = ss.str();
        llvm::StringRef desc_ref = desc;
        if (desc_ref.consume_front("is equal to \"") &&
            desc_ref.consume_back("\"")) {
          *os << desc_ref.str();
        } else {
          *os << desc;
        }
        *os << "\n";
      }
    }
  }
  *os << "=== diff end\n";
}

}  // namespace Carbon::Testing

#endif  // CARBON_TESTING_BASE_UNIFIED_DIFF_H_
