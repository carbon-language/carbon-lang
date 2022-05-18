// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_TEST_HELPERS_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_TEST_HELPERS_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/check.h"
#include "common/ostream.h"
#include "llvm/ADT/StringExtras.h"
#include "toolchain/semantics/nodes/declared_name_test_matchers.h"
#include "toolchain/semantics/nodes/function_test_matchers.h"
#include "toolchain/semantics/nodes/infix_operator_test_matchers.h"
#include "toolchain/semantics/nodes/literal_test_matchers.h"
#include "toolchain/semantics/nodes/pattern_binding_test_matchers.h"
#include "toolchain/semantics/nodes/return_test_matchers.h"
#include "toolchain/semantics/semantics_ir_for_test.h"

namespace Carbon::Testing {

inline auto MappedNode(::testing::Matcher<std::string> key,
                       ::testing::Matcher<Semantics::Declaration> value)
    -> ::testing::Matcher<llvm::StringMapEntry<Semantics::Declaration>> {
  return ::testing::AllOf(
      ::testing::Property(
          "key", &llvm::StringMapEntry<Semantics::Declaration>::getKey, key),
      ::testing::Property(
          "value", &llvm::StringMapEntry<Semantics::Declaration>::getValue,
          value));
}

// Avoids gtest confusion of how to print llvm::None.
MATCHER(IsNone, "is llvm::None") { return arg == llvm::None; }

inline auto StatementBlock(
    ::testing::Matcher<llvm::ArrayRef<Semantics::Statement>> nodes_matcher,
    ::testing::Matcher<llvm::StringMap<Semantics::Statement>>
        name_lookup_matcher) -> ::testing::Matcher<Semantics::StatementBlock> {
  return ::testing::AllOf(
      ::testing::Property("nodes", &Semantics::StatementBlock::nodes,
                          nodes_matcher),
      ::testing::Property("name_lookup",
                          &Semantics::StatementBlock::name_lookup,
                          name_lookup_matcher));
}

}  // namespace Carbon::Testing

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_TEST_HELPERS_H_
