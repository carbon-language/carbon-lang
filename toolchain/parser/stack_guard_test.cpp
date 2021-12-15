// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/stack_guard.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace Carbon {
namespace {

using ::testing::HasSubstr;

static void Recurse(StackGuard guard, int depth) {
  // Should fail on StackGuard, but this silences -Winfinite-recursion.
  ASSERT_LE(depth, 1000);
  Recurse(guard, depth + 1);
}

TEST(StackGuardTest, Guarding) {
  StackGuard guard = StackGuard::Root();
  EXPECT_DEATH({ Recurse(guard, 0); }, HasSubstr("Exceeded recursion limit"));
}

}  // namespace
}  // namespace Carbon
