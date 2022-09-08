// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_SET_NAME_TEST_MATCHERS_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_SET_NAME_TEST_MATCHERS_H_

#include <gmock/gmock.h>

#include "llvm/ADT/StringExtras.h"
#include "toolchain/semantics/nodes/set_name.h"
#include "toolchain/semantics/semantics_ir_for_test.h"

namespace Carbon::Testing {

MATCHER_P2(
    SetName, name_matcher, target_id_matcher,
    llvm::formatv(
        "SetName(`{0}`, `{1}`)",
        ::testing::DescribeMatcher<llvm::StringRef>(name_matcher),
        ::testing::DescribeMatcher<Semantics::NodeId>(target_id_matcher))) {
  const Semantics::NodeRef& node_ref = arg;
  if (auto node = SemanticsIRForTest::GetNode<Semantics::SetName>(node_ref)) {
    return ExplainMatchResult(name_matcher, node->name(), result_listener) &&
           ExplainMatchResult(target_id_matcher, node->target_id(),
                              result_listener);
  } else {
    *result_listener << "node is not a SetName";
    return result_listener;
  }
}

}  // namespace Carbon::Testing

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_SET_NAME_TEST_MATCHERS_H_
