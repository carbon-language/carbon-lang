// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_FUNCTION_TEST_MATCHERS_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_FUNCTION_TEST_MATCHERS_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/ADT/StringExtras.h"
#include "toolchain/semantics/nodes/function.h"
#include "toolchain/semantics/nodes/pattern_binding.h"
#include "toolchain/semantics/semantics_ir_for_test.h"

namespace Carbon::Testing {

MATCHER_P(FunctionName, name_matcher,
          llvm::formatv("fn `{0}`", ::testing::DescribeMatcher<llvm::StringRef>(
                                        name_matcher))) {
  const Semantics::Declaration& decl = arg;
  if (auto function =
          SemanticsIRForTest::GetDeclaration<Semantics::Function>(decl)) {
    return ExplainMatchResult(
        name_matcher, SemanticsIRForTest::GetNodeText(function->name().node()),
        result_listener);
  } else {
    *result_listener << "node is not a function";
    return result_listener;
  }
}

MATCHER_P4(
    Function, name_matcher, param_matcher, return_matcher, body_matcher,
    llvm::formatv(
        "fn `{0}` params `{1}` returns `{2}` body `{3}`",
        ::testing::DescribeMatcher<llvm::StringRef>(name_matcher),
        ::testing::DescribeMatcher<llvm::ArrayRef<Semantics::PatternBinding>>(
            param_matcher),
        ::testing::DescribeMatcher<llvm::Optional<Semantics::Expression>>(
            return_matcher),
        ::testing::DescribeMatcher<Semantics::StatementBlock>(body_matcher))) {
  const Semantics::Declaration& decl = arg;
  if (auto function =
          SemanticsIRForTest::GetDeclaration<Semantics::Function>(decl)) {
    return ExplainMatchResult(
               name_matcher,
               SemanticsIRForTest::GetNodeText(function->name().node()),
               result_listener) &&
           ExplainMatchResult(param_matcher, function->params(),
                              result_listener) &&
           ExplainMatchResult(return_matcher, function->return_expr(),
                              result_listener) &&
           ExplainMatchResult(body_matcher, function->body(), result_listener);
  } else {
    *result_listener << "node is not a function";
    return result_listener;
  }
}

}  // namespace Carbon::Testing

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_FUNCTION_TEST_MATCHERS_H_
