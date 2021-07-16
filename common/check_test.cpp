// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/check.h"

#include "gtest/gtest.h"

namespace Carbon {

TEST(CheckTest, CheckTrue) { CHECK(true); }

TEST(CheckTest, CheckFalse) {
  ASSERT_DEATH({ CHECK(false); }, "CHECK failure: false");
}

TEST(CheckTest, CheckTrueMessage) { CHECK(true) << "msg"; }

TEST(CheckTest, CheckFalseMessage) {
  ASSERT_DEATH({ CHECK(false) << "msg"; }, "CHECK failure: false");
}

}  // namespace Carbon
