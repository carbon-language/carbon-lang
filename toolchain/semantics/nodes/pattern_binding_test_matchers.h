// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_PATTERN_BINDING_TEST_MATCHERS_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_PATTERN_BINDING_TEST_MATCHERS_H_

#include <gtest/gtest.h>

#include "llvm/ADT/StringExtras.h"
#include "toolchain/semantics/nodes/declared_name_test_matchers.h"
#include "toolchain/semantics/nodes/pattern_binding.h"
#include "toolchain/semantics/semantics_ir_for_test.h"

namespace Carbon::Testing {

inline auto PatternBinding(
    ::testing::Matcher<llvm::StringRef> name_matcher,
    ::testing::Matcher<Semantics::Expression> type_matcher)
    -> ::testing::Matcher<Semantics::PatternBinding> {
  return ::testing::AllOf(
      ::testing::Property("name", &Semantics::PatternBinding::name,
                          DeclaredName(name_matcher)),
      ::testing::Property("type", &Semantics::PatternBinding::type,
                          type_matcher));
}

}  // namespace Carbon::Testing

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_PATTERN_BINDING_TEST_MATCHERS_H_
