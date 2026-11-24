// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/env.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>

namespace Carbon {
namespace {

using ::testing::Eq;
using ::testing::StrEq;

TEST(EnvTest, SetAndUnset) {
  constexpr const char* TestVar = "CARBON_ENV_TEST_VAR";
  UnsetEnv(TestVar);
  EXPECT_THAT(std::getenv(TestVar), Eq(nullptr));

  SetEnv(TestVar, "initial");
  ASSERT_NE(std::getenv(TestVar), nullptr);
  EXPECT_THAT(std::getenv(TestVar), StrEq("initial"));

  // Non-overwriting set preserves existing value.
  SetEnv(TestVar, "ignored", /*overwrite=*/false);
  ASSERT_NE(std::getenv(TestVar), nullptr);
  EXPECT_THAT(std::getenv(TestVar), StrEq("initial"));

  // Overwriting set replaces existing value.
  SetEnv(TestVar, "updated", /*overwrite=*/true);
  ASSERT_NE(std::getenv(TestVar), nullptr);
  EXPECT_THAT(std::getenv(TestVar), StrEq("updated"));

  UnsetEnv(TestVar);
  EXPECT_THAT(std::getenv(TestVar), Eq(nullptr));

  // Non-overwriting set sets the variable when it is unset.
  SetEnv(TestVar, "fresh", /*overwrite=*/false);
  ASSERT_NE(std::getenv(TestVar), nullptr);
  EXPECT_THAT(std::getenv(TestVar), StrEq("fresh"));

  UnsetEnv(TestVar);
  EXPECT_THAT(std::getenv(TestVar), Eq(nullptr));
}

}  // namespace
}  // namespace Carbon
