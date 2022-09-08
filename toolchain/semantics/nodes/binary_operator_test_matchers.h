// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_BINARY_OPERATOR_TEST_MATCHERS_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_BINARY_OPERATOR_TEST_MATCHERS_H_

#include <gmock/gmock.h>

#include "llvm/ADT/StringExtras.h"
#include "toolchain/semantics/nodes/binary_operator.h"
#include "toolchain/semantics/semantics_ir_for_test.h"

namespace Carbon::Testing {

MATCHER_P4(
    BinaryOperator, id_matcher, op_matcher, lhs_id_matcher, rhs_id_matcher,
    llvm::formatv(
        "BinaryOperator(`{0}`, `{1}`, `{2}`, `{3}`)",
        ::testing::DescribeMatcher<Semantics::NodeId>(id_matcher),
        ::testing::DescribeMatcher<Semantics::BinaryOperator::Op>(op_matcher),
        ::testing::DescribeMatcher<Semantics::NodeId>(lhs_id_matcher),
        ::testing::DescribeMatcher<Semantics::NodeId>(rhs_id_matcher))) {
  const Semantics::NodeRef& node_ref = arg;
  if (auto op =
          SemanticsIRForTest::GetNode<Semantics::BinaryOperator>(node_ref)) {
    return ExplainMatchResult(id_matcher, op->id(), result_listener) &&
           ExplainMatchResult(op_matcher, op->op(), result_listener) &&
           ExplainMatchResult(lhs_id_matcher, op->lhs_id(), result_listener) &&
           ExplainMatchResult(rhs_id_matcher, op->rhs_id(), result_listener);
  } else {
    *result_listener << "node is not a BinaryOperator";
    return result_listener;
  }
}

}  // namespace Carbon::Testing

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_BINARY_OPERATOR_TEST_MATCHERS_H_
