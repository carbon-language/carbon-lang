// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_INTEGER_LITERAL_TEST_MATCHERS_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_INTEGER_LITERAL_TEST_MATCHERS_H_

#include <gmock/gmock.h>

#include "llvm/ADT/StringExtras.h"
#include "toolchain/semantics/nodes/integer_literal.h"
#include "toolchain/semantics/semantics_ir_for_test.h"

namespace Carbon::Testing {

MATCHER_P2(
    IntegerLiteral, id_matcher, value_matcher,
    llvm::formatv("IntegerLiteral(`{0}`, `{1}`)",
                  ::testing::DescribeMatcher<Semantics::NodeId>(id_matcher),
                  ::testing::DescribeMatcher<llvm::APInt>(value_matcher))) {
  const Semantics::NodeRef& node_ref = arg;
  if (auto lit =
          SemanticsIRForTest::GetNode<Semantics::IntegerLiteral>(node_ref)) {
    return ExplainMatchResult(id_matcher, lit->id(), result_listener) &&
           ExplainMatchResult(value_matcher, lit->value(), result_listener);
  } else {
    *result_listener << "node is not a IntegerLiteral";
    return result_listener;
  }
}

}  // namespace Carbon::Testing

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_INTEGER_LITERAL_TEST_MATCHERS_H_
