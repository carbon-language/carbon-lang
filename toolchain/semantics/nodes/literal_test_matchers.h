// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_LITERAL_TEST_MATCHERS_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_LITERAL_TEST_MATCHERS_H_

#include <gtest/gtest.h>

#include "llvm/ADT/StringExtras.h"
#include "toolchain/semantics/nodes/literal.h"
#include "toolchain/semantics/semantics_ir_for_test.h"

namespace Carbon::Testing {

MATCHER_P(
    Literal, text_matcher,
    llvm::formatv("Literal {0}",
                  ::testing::DescribeMatcher<llvm::StringRef>(text_matcher))) {
  const Semantics::Expression& expr = arg;
  if (auto lit = SemanticsIRForTest::GetExpression<Semantics::Literal>(expr)) {
    return ExplainMatchResult(text_matcher,
                              SemanticsIRForTest::GetNodeText(lit->node()),
                              result_listener);
  } else {
    *result_listener << "node is not a literal";
    return result_listener;
  }
}

}  // namespace Carbon::Testing

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_LITERAL_TEST_MATCHERS_H_
