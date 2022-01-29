// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_SYNTAX_PARSE_TEST_MATCHERS_H_
#define EXECUTABLE_SEMANTICS_SYNTAX_PARSE_TEST_MATCHERS_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "executable_semantics/syntax/parse_test_matchers_internal.h"

namespace Carbon {

// Matches the return value of `Parse()` if it represents a successful parse
// whose output matches the given matcher.
inline auto ParsedAs(::testing::Matcher<AST> ast_matcher) {
  return TestingInternal::ParsedAsMatcher(std::move(ast_matcher));
}

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_SYNTAX_PARSE_TEST_MATCHERS_H_
