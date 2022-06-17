// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_INFIX_OPERATOR_TEST_MATCHERS_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_INFIX_OPERATOR_TEST_MATCHERS_H_

#include <gtest/gtest.h>

#include "llvm/ADT/StringExtras.h"
#include "toolchain/semantics/nodes/infix_operator.h"
#include "toolchain/semantics/semantics_ir_for_test.h"

namespace Carbon::Testing {

MATCHER_P3(
    InfixOperator, lhs_matcher, op_matcher, rhs_matcher,
    llvm::formatv(
        "InfixOperator {0} {1} {2}",
        ::testing::DescribeMatcher<Semantics::Expression>(lhs_matcher),
        ::testing::DescribeMatcher<llvm::StringRef>(op_matcher),
        ::testing::DescribeMatcher<Semantics::Expression>(rhs_matcher))) {
  const Semantics::Expression& expr = arg;
  if (auto infix =
          SemanticsIRForTest::GetExpression<Semantics::InfixOperator>(expr)) {
    return ExplainMatchResult(op_matcher,
                              SemanticsIRForTest::GetNodeText(infix->node()),
                              result_listener) &&
           ExplainMatchResult(lhs_matcher, infix->lhs(), result_listener) &&
           ExplainMatchResult(rhs_matcher, infix->rhs(), result_listener);
  } else {
    *result_listener << "node is not a literal";
    return result_listener;
  }
}

}  // namespace Carbon::Testing

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_INFIX_OPERATOR_TEST_MATCHERS_H_
