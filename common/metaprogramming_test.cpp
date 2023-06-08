// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/metaprogramming.h"

#include <gtest/gtest.h>

#include <string>

namespace Carbon::Testing {
namespace {

TEST(MetaProgrammingTest, RequiresTest) {
  auto lambda = [](int a, int b) { return a + b; };

  bool result = Requires<int, int>(lambda);
  EXPECT_TRUE(result);
}

}  // namespace
}  // namespace Carbon::Testing