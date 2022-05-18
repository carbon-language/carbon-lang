// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_RETURN_TEST_MATCHERS_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_RETURN_TEST_MATCHERS_H_

#include <gtest/gtest.h>

#include "llvm/ADT/StringExtras.h"
#include "toolchain/semantics/nodes/return.h"
#include "toolchain/semantics/semantics_ir_for_test.h"

namespace Carbon::Testing {

MATCHER_P(Return, expr_matcher,
          llvm::formatv(
              "Return {0}",
              ::testing::DescribeMatcher<llvm::Optional<Semantics::Expression>>(
                  expr_matcher))) {
  const Semantics::Statement& stmt = arg;
  if (auto ret = SemanticsIRForTest::GetStatement<Semantics::Return>(stmt)) {
    return ExplainMatchResult(expr_matcher, ret->expression(), result_listener);
  } else {
    *result_listener << "node is not a function";
    return result_listener;
  }
}

}  // namespace Carbon::Testing

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_RETURN_TEST_MATCHERS_H_
