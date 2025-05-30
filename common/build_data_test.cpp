// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/build_data.h"

#include <gmock/gmock.h>

#include "common/ostream.h"

namespace Carbon {
namespace {

using ::testing::EndsWith;
using ::testing::HasSubstr;

TEST(BuildDataTest, Values) {
  EXPECT_FALSE(BuildData::Platform.empty());
  // Can't really test BuildCoverageEnabled.

  // This doesn't require `//`, for a bit of incremental robustness.
  EXPECT_THAT(BuildData::TargetName, EndsWith("/common:build_data_test"));

  // This doesn't examine the path, for platform robustness (e.g. `.exe`).
  EXPECT_THAT(BuildData::BuildTarget, HasSubstr("build_data_test"));
}

}  // namespace
}  // namespace Carbon
