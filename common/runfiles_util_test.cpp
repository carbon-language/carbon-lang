// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/runfiles_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>

namespace Carbon::Testing {
namespace {

TEST(RunfilesUtilTest, GetRunfilesDir) {
  EXPECT_THAT(GetRunfilesDir(),
              testing::EndsWith("/common/runfiles_util_test.runfiles"));
}

}  // namespace
}  // namespace Carbon::Testing
