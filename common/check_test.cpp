// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/check.h"

#include "gtest/gtest.h"

namespace Carbon {

TEST(CheckTest, CheckTrue) { CHECK(true); }

TEST(CheckTest, CheckFalse) {
  ASSERT_DEATH({ CHECK(false); }, "CHECK failure: false\n");
}

TEST(CheckTest, CheckTrueCallbackNotUsed) {
  bool called = false;
  auto callback = [&]() {
    called = true;
    return "called";
  };
  CHECK(true) << callback();
  EXPECT_FALSE(called);
}

TEST(CheckTest, CheckFalseMessage) {
  ASSERT_DEATH({ CHECK(false) << "msg"; }, "CHECK failure: false: msg\n");
}

TEST(CheckTest, CheckOutputForms) {
  const char msg[] = "msg";
  std::string str = "str";
  int i = 1;
  CHECK(true) << msg << str << i << 0;
  ASSERT_DEATH({ CHECK(false) << msg << str << i; },
               "CHECK failure: false: msgstr1\n");
}

TEST(CheckTest, Fatal) {
  ASSERT_DEATH({ FATAL_IF(true) << "msg"; }, "FATAL: msg\n");
}

TEST(CheckTest, FatalNoInput) {
  ASSERT_DEATH({ FATAL_IF(true); }, "FATAL: \n");
}

}  // namespace Carbon
