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

  template <typename T, typename ErrorT>
  auto MatchAndExplain(const ErrorOr<T, ErrorT>& result,
                       ::testing::MatchResultListener* listener) const -> bool {
    if (result.ok()) {
      *listener << "is a success";
      return false;
    } else {
      RawStringOstream os;
      os << result.error();
      return matcher_.MatchAndExplain(os.TakeStr(), listener);
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

// Implementation of a success matcher for a specific `T` and `ErrorT` in an
// `ErrorOr`. Supports a nested matcher for the `T` value.
template <typename T, typename ErrorT>
class IsSuccessMatcherImpl
    : public ::testing::MatcherInterface<const ErrorOr<T, ErrorT>&> {
 public:
  explicit IsSuccessMatcherImpl(const ::testing::Matcher<T>& matcher)
      : matcher_(matcher) {}

  auto MatchAndExplain(const ErrorOr<T, ErrorT>& result,
                       ::testing::MatchResultListener* listener) const
      -> bool override {
    if (result.ok()) {
      return matcher_.MatchAndExplain(*result, listener);
    } else {
      *listener << "is an error with `" << result.error() << "`";
      return false;
    }
  }

  auto DescribeTo(std::ostream* os) const -> void override {
    *os << "is a success and matches ";
    matcher_.DescribeTo(os);
  }

  auto DescribeNegationTo(std::ostream* os) const -> void override {
    *os << "is an error or does not match ";
    matcher_.DescribeNegationTo(os);
  }

 private:
  ::testing::Matcher<T> matcher_;
};

// Polymorphic match implementation for GoogleTest.
//
// To support matching arbitrary types that `InnerMatcher` can also match, this
// itself must match arbitrary types. This is accomplished by not being a
// matcher itself, but by being convertible into matchers for any particular
// `ErrorOr`.
template <typename InnerMatcher>
class IsSuccessMatcher {
 public:
  explicit IsSuccessMatcher(InnerMatcher matcher)
      : matcher_(std::move(matcher)) {}

  template <typename T, typename ErrorT>
  explicit(false)
  // NOLINTNEXTLINE(google-explicit-constructor): Required for matcher APIs.
  operator ::testing::Matcher<const ErrorOr<T, ErrorT>&>() const {
    return ::testing::Matcher<const ErrorOr<T, ErrorT>&>(
        new IsSuccessMatcherImpl<T, ErrorT>(
            ::testing::SafeMatcherCast<T>(matcher_)));
  }

 private:
  InnerMatcher matcher_;
};

// Returns a matcher the value for a non-error state of `ErrorOr<T>`.
//
// For example:
//   EXPECT_THAT(my_result, IsSuccess(Eq(3)));
template <typename InnerMatcher>
auto IsSuccess(InnerMatcher matcher) -> IsSuccessMatcher<InnerMatcher> {
  return IsSuccessMatcher<InnerMatcher>(matcher);
}

}  // namespace Carbon::Testing

namespace Carbon {

// Supports printing `ErrorOr<T>` to `std::ostream` in tests.
template <typename T, typename ErrorT>
auto operator<<(std::ostream& out, const ErrorOr<T, ErrorT>& error_or)
    -> std::ostream& {
  if (error_or.ok()) {
    // Try and print the value, but only if we can find a viable `<<` overload
    // for the value type. This should ensure that the `formatv` below can
    // compile cleanly, and avoid erroring when using matchers on `ErrorOr` with
    // unprintable value types.
    if constexpr (requires(const T& value) { out << value; }) {
      out << llvm::formatv("ErrorOr{{.value = `{0}`}}", *error_or);
    } else {
      out << "ErrorOr{{.value = `<unknown>`}}";
    }
  } else {
    out << llvm::formatv("ErrorOr{{.error = \"{0}\"}}", error_or.error());
  }
  return out;
}

}  // namespace Carbon

#endif  // CARBON_COMMON_ERROR_TEST_HELPERS_H_
