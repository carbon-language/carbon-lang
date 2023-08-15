// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/common/error_builders.h"

#include <gtest/gtest.h>

#include "common/testing/base/test_raw_ostream.h"
#include "explorer/common/source_location.h"

namespace Carbon::Testing {
namespace {

TEST(ErrorBuildersTest, ProgramError) {
  Error err = ProgramError(SourceLocation("x", 1, FileKind::Main)) << "test";
  EXPECT_EQ(err.location(), "x:1");
  EXPECT_EQ(err.message(), "test");

  TestRawOstream out;
  out << err;
  EXPECT_EQ(out.TakeStr(), "x:1: test");
}

}  // namespace
}  // namespace Carbon::Testing
