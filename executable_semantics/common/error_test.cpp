// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/common/error.h"

#include "gtest/gtest.h"

namespace Carbon {

TEST(ErrorTest, UserErrorIfTrue) {
  ASSERT_DEATH({ USER_ERROR_IF(true) << "test"; }, "ERROR: test\n");
}

TEST(ErrorTest, UserErrorIfFalse) {
  bool called = false;
  auto callback = [&]() {
    called = true;
    return "called";
  };
  USER_ERROR_IF(false) << callback();
  EXPECT_FALSE(called);
}

}  // namespace Carbon
