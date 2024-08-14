// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/base/global_exe_path.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/Support/FileSystem.h"

namespace Carbon::Testing {
namespace {

using ::testing::StrNe;

TEST(TestExePathTest, Test) {
  llvm::StringRef exe_path = GetExePath();
  EXPECT_THAT(exe_path, StrNe(""));
  EXPECT_TRUE(llvm::sys::fs::exists(exe_path));
  EXPECT_TRUE(llvm::sys::fs::can_execute(exe_path));
}

}  // namespace
}  // namespace Carbon::Testing
