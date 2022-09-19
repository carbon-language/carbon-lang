// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/check.h"

#include <gtest/gtest.h>

namespace Carbon::Testing {
namespace {

TEST(CheckTest, CheckTrue) { CARBON_CHECK(true); }

TEST(CheckTest, CheckFalse) {
  ASSERT_DEATH({ CARBON_CHECK(false); },
               "Stack trace:\n"
               "(.|\n)+\n"
               "CHECK failure at common/check_test.cpp:\\d+: false\n");
}

TEST(CheckTest, CheckTrueCallbackNotUsed) {
  bool called = false;
  auto callback = [&]() {
    called = true;
    return "called";
  };
  CARBON_CHECK(true) << callback();
  EXPECT_FALSE(called);
}

TEST(CheckTest, CheckFalseMessage) {
  ASSERT_DEATH({ CARBON_CHECK(false) << "msg"; },
               "CHECK failure at common/check_test.cpp:.+: false: msg\n");
}

TEST(CheckTest, CheckOutputForms) {
  const char msg[] = "msg";
  std::string str = "str";
  int i = 1;
  CARBON_CHECK(true) << msg << str << i << 0;
}

TEST(CheckTest, Fatal) {
  ASSERT_DEATH({ CARBON_FATAL() << "msg"; },
               "FATAL failure at common/check_test.cpp:.+: msg\n");
}

auto FatalNoReturnRequired() -> int { CARBON_FATAL() << "msg"; }

TEST(ErrorTest, FatalNoReturnRequired) {
  ASSERT_DEATH({ FatalNoReturnRequired(); },
               "FATAL failure at common/check_test.cpp:.+: msg\n");
}

}  // namespace
}  // namespace Carbon::Testing
