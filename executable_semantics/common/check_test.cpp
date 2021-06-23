// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/common/check.h"

#include "gtest/gtest.h"

namespace Carbon {

TEST(CheckTest, CheckTrue) { CHECK(true); }

TEST(CheckTest, CheckFalse) {
  ASSERT_DEATH({ CHECK(false); }, "LLVM ERROR: CHECK failure: false");
}

}  // namespace Carbon
