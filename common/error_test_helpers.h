// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_ERROR_TEST_HELPERS_H_
#define CARBON_COMMON_ERROR_TEST_HELPERS_H_

#include <gmock/gmock.h>

#include "common/error.h"

namespace Carbon::Testing {

// Matches the message for an error state of `ErrorOr<T>`. For example:
//   EXPECT_THAT(my_result, IsError(StrEq("error message")));
class IsError {
 public:
  // NOLINTNEXTLINE(readability-identifier-naming)
  using is_gtest_matcher = void;

  explicit IsError(::testing::Matcher<std::string> matcher)
      : matcher_(std::move(matcher)) {}

  template <typename T>
  auto MatchAndExplain(const ErrorOr<T>& result,
                       ::testing::MatchResultListener* listener) const -> bool {
    if (result.ok()) {
      *listener->stream() << "is a success";
      return false;
    } else {
      return matcher_.MatchAndExplain(result.error().message(), listener);
    }
  }

  auto DescribeTo(std::ostream* os) const -> void {
    *os << "is an error and matches ";
    matcher_.DescribeTo(os);
  }

  auto DescribeNegationTo(std::ostream* os) const -> void {
    *os << "is a success or does not match ";
    matcher_.DescribeTo(os);
  }

 private:
  ::testing::Matcher<std::string> matcher_;
};

// Matches the value for a non-error state of `ErrorOr<T>`. For example:
//   EXPECT_THAT(my_result, IsSuccess(Eq(3)));
template <typename InnerMatcher>
class IsSuccessMatcher {
 public:
  // NOLINTNEXTLINE(readability-identifier-naming)
  using is_gtest_matcher = void;

  explicit IsSuccessMatcher(InnerMatcher matcher)
      : matcher_(std::move(matcher)) {}

  template <typename T>
  auto MatchAndExplain(const ErrorOr<T>& result,
                       ::testing::MatchResultListener* listener) const -> bool {
    if (result.ok()) {
      return ::testing::Matcher<T>(matcher_).MatchAndExplain(*result, listener);
    } else {
      *listener->stream() << "is an error with `" << result.error().message()
                          << "`";
      return false;
    }
  }

  auto DescribeTo(std::ostream* os) const -> void {
    *os << "is a success and matches ";
    matcher_.DescribeTo(os);
  }

  auto DescribeNegationTo(std::ostream* os) const -> void {
    *os << "is an error or does not match ";
    matcher_.DescribeTo(os);
  }

 private:
  InnerMatcher matcher_;
};

// Wraps `IsSuccessMatcher` for the inner matcher deduction.
template <typename InnerMatcher>
auto IsSuccess(InnerMatcher matcher) -> IsSuccessMatcher<InnerMatcher> {
  return IsSuccessMatcher<InnerMatcher>(matcher);
}

}  // namespace Carbon::Testing

namespace Carbon {

// Supports printing `ErrorOr<T>` to `std::ostream` in tests.
template <typename T>
auto operator<<(std::ostream& out, const ErrorOr<T>& error_or)
    -> std::ostream& {
  if (error_or.ok()) {
    out << llvm::formatv("ErrorOr{{.value = `{0}`}}", *error_or);
  } else {
    out << llvm::formatv("ErrorOr{{.error = \"{0}\"}}",
                         error_or.error().message());
  }
  return out;
}

}  // namespace Carbon

#endif  // CARBON_COMMON_ERROR_TEST_HELPERS_H_
