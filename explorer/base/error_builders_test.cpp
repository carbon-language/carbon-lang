// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/base/error_builders.h"

#include <gtest/gtest.h>

#include "explorer/base/source_location.h"
#include "testing/base/test_raw_ostream.h"

namespace Carbon {
namespace {

TEST(ErrorBuildersTest, ProgramError) {
  Error err = ProgramError(SourceLocation("x", 1, FileKind::Main)) << "test";
  EXPECT_EQ(err.location(), "x:1");
  EXPECT_EQ(err.message(), "test");

  Testing::TestRawOstream out;
  out << err;
  EXPECT_EQ(out.TakeStr(), "x:1: test");
}

}  // namespace
}  // namespace Carbon
