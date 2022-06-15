// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_DECLARED_NAME_TEST_MATCHERS_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_DECLARED_NAME_TEST_MATCHERS_H_

#include <gtest/gtest.h>

#include "llvm/ADT/StringExtras.h"
#include "toolchain/semantics/nodes/declared_name.h"
#include "toolchain/semantics/semantics_ir_for_test.h"

namespace Carbon::Testing {

MATCHER_P(
    DeclaredName, name_matcher,
    llvm::formatv("DeclaredName {0}",
                  ::testing::DescribeMatcher<llvm::StringRef>(name_matcher))) {
  const Semantics::DeclaredName& name = arg;
  return ExplainMatchResult(name_matcher,
                            SemanticsIRForTest::GetNodeText(name.node()),
                            result_listener);
}

}  // namespace Carbon::Testing

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_DECLARED_NAME_TEST_MATCHERS_H_
