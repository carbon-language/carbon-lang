// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_SYNTAX_PARSE_TEST_MATCHERS_INTERNAL_H_
#define CARBON_EXPLORER_SYNTAX_PARSE_TEST_MATCHERS_INTERNAL_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ostream>
#include <variant>

#include "explorer/syntax/parse.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon::TestingInternal {

// Implementation of ParsedAs(). See there for detailed documentation.
class ParsedAsMatcher {
 public:
  // NOLINTNEXTLINE(readability-identifier-naming)
  using is_gtest_matcher = void;

  explicit ParsedAsMatcher(::testing::Matcher<AST> ast_matcher)
      : ast_matcher_(std::move(ast_matcher)) {}

  void DescribeTo(std::ostream* out) const {
    DescribeToImpl(out, /*negated=*/false);
  }

  void DescribeNegationTo(std::ostream* out) const {
    DescribeToImpl(out, /*negated=*/true);
  }

  auto MatchAndExplain(const ErrorOr<AST>& result,
                       ::testing::MatchResultListener* listener) const -> bool {
    if (!result.ok()) {
      *listener << "is a failed parse with error: " << result.error().message();
      return false;
    } else {
      *listener << "is a successful parse whose ";
      return ast_matcher_.MatchAndExplain(*result, listener);
    }
  }

 private:
  void DescribeToImpl(std::ostream* out, bool negated) const {
    *out << "is " << (negated ? "not " : "")
         << "a successful parse result whose ";
    ast_matcher_.DescribeTo(out);
  }

  ::testing::Matcher<AST> ast_matcher_;
};

}  // namespace Carbon::TestingInternal

#endif  // CARBON_EXPLORER_SYNTAX_PARSE_TEST_MATCHERS_INTERNAL_H_
