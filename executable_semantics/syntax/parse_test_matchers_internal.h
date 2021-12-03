// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_SYNTAX_PARSE_TEST_MATCHERS_INTERNAL_H_
#define EXECUTABLE_SEMANTICS_SYNTAX_PARSE_TEST_MATCHERS_INTERNAL_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ostream>
#include <variant>

#include "executable_semantics/syntax/parse.h"

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

  auto MatchAndExplain(const std::variant<AST, SyntaxErrorCode>& result,
                       ::testing::MatchResultListener* listener) const -> bool {
    if (std::holds_alternative<SyntaxErrorCode>(result)) {
      *listener << "holds error code " << std::get<SyntaxErrorCode>(result);
      return false;
    } else {
      *listener << "is a successful parse whose ";
      return ast_matcher_.MatchAndExplain(std::get<AST>(result), listener);
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

#endif  // EXECUTABLE_SEMANTICS_SYNTAX_PARSE_TEST_MATCHERS_INTERNAL_H_
