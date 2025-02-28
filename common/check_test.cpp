// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/check.h"

#include <gtest/gtest.h>

namespace Carbon {
namespace {

// Non-constexpr functions that always return true and false, to bypass constant
// condition checking.
auto AlwaysTrue() -> bool { return true; }
auto AlwaysFalse() -> bool { return false; }

TEST(CheckTest, CheckTrue) { CARBON_CHECK(AlwaysTrue()); }

TEST(CheckTest, CheckFalse) {
  ASSERT_DEATH({ CARBON_CHECK(AlwaysFalse()); },
               R"(
CHECK failure at common/check_test.cpp:\d+: AlwaysFalse\(\)
)");
}

TEST(CheckTest, CheckFalseHasStackDump) {
  ASSERT_DEATH({ CARBON_CHECK(AlwaysFalse()); }, "\nStack dump:\n");
}

TEST(CheckTest, CheckTrueCallbackNotUsed) {
  bool called = false;
  auto callback = [&]() {
    called = true;
    return "called";
  };
  CARBON_CHECK(AlwaysTrue(), "{0}", callback());
  EXPECT_FALSE(called);
}

TEST(CheckTest, CheckFalseMessage) {
  ASSERT_DEATH({ CARBON_CHECK(AlwaysFalse(), "msg"); },
               R"(
CHECK failure at common/check_test.cpp:.+: AlwaysFalse\(\): msg
)");
}

TEST(CheckTest, CheckFalseFormattedMessage) {
  const char msg[] = "msg";
  std::string str = "str";
  int i = 1;
  ASSERT_DEATH(
      { CARBON_CHECK(AlwaysFalse(), "{0} {1} {2} {3}", msg, str, i, 0); },
      R"(
CHECK failure at common/check_test.cpp:.+: AlwaysFalse\(\): msg str 1 0
)");
}

TEST(CheckTest, CheckOutputForms) {
  const char msg[] = "msg";
  std::string str = "str";
  int i = 1;
  CARBON_CHECK(AlwaysTrue(), "{0} {1} {2} {3}", msg, str, i, 0);
}

TEST(CheckTest, Fatal) {
  ASSERT_DEATH({ CARBON_FATAL("msg"); },
               "\nFATAL failure at common/check_test.cpp:.+: msg\n");
}

TEST(CheckTest, FatalHasStackDump) {
  ASSERT_DEATH({ CARBON_FATAL("msg"); }, "\nStack dump:\n");
}

auto FatalNoReturnRequired() -> int { CARBON_FATAL("msg"); }

TEST(ErrorTest, FatalNoReturnRequired) {
  ASSERT_DEATH({ FatalNoReturnRequired(); },
               "\nFATAL failure at common/check_test.cpp:.+: msg\n");
}

// Detects whether `CARBON_CHECK(F())` compiles.
template <auto F>
concept CheckCompilesWithCondition = requires { CARBON_CHECK(F()); };

TEST(CheckTest, CheckConstantCondition) {
  EXPECT_TRUE(CheckCompilesWithCondition<[] { return AlwaysTrue(); }>);
  EXPECT_TRUE(CheckCompilesWithCondition<[] { return AlwaysFalse(); }>);
  EXPECT_FALSE(CheckCompilesWithCondition<[] { return true; }>);
  EXPECT_FALSE(CheckCompilesWithCondition<[] { return false; }>);
}

}  // namespace
}  // namespace Carbon
