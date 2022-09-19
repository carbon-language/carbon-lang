// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_FUNCTION_TEST_MATCHERS_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_FUNCTION_TEST_MATCHERS_H_

#include <gmock/gmock.h>

#include "llvm/ADT/StringExtras.h"
#include "toolchain/semantics/nodes/function.h"
#include "toolchain/semantics/semantics_ir_for_test.h"

namespace Carbon::Testing {

MATCHER_P2(Function, id_matcher, body_matcher,
           llvm::formatv(
               "Function(`{0}`, `{1}`)",
               ::testing::DescribeMatcher<Semantics::NodeId>(id_matcher),
               ::testing::DescribeMatcher<llvm::ArrayRef<Semantics::NodeRef>>(
                   body_matcher))) {
  const Semantics::NodeRef& node_ref = arg;
  if (auto function =
          SemanticsIRForTest::GetNode<Semantics::Function>(node_ref)) {
    return ExplainMatchResult(id_matcher, function->id(), result_listener) &&
           ExplainMatchResult(body_matcher, function->body(), result_listener);
  } else {
    *result_listener << "node is not a Function";
    return result_listener;
  }
}

}  // namespace Carbon::Testing

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_FUNCTION_TEST_MATCHERS_H_
