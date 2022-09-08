// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_RETURN_TEST_MATCHERS_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_RETURN_TEST_MATCHERS_H_

#include <gmock/gmock.h>

#include "llvm/ADT/StringExtras.h"
#include "toolchain/semantics/nodes/return.h"
#include "toolchain/semantics/semantics_ir_for_test.h"

namespace Carbon::Testing {

MATCHER_P(
    Return, target_id_matcher,
    llvm::formatv("Return(`{0}`)",
                  ::testing::DescribeMatcher<llvm::Optional<Semantics::NodeId>>(
                      target_id_matcher))) {
  const Semantics::NodeRef& node_ref = arg;
  if (auto ret = SemanticsIRForTest::GetNode<Semantics::Return>(node_ref)) {
    return ExplainMatchResult(target_id_matcher, ret->target_id(),
                              result_listener);
  } else {
    *result_listener << "node is not a Return";
    return result_listener;
  }
}

}  // namespace Carbon::Testing

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_RETURN_TEST_MATCHERS_H_
