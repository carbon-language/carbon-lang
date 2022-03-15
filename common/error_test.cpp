// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/error.h"

#include <gtest/gtest.h>

namespace Carbon::Testing {
namespace {

TEST(ErrorTest, Error) {
  Error err("test");
  EXPECT_FALSE(err.ok());
  EXPECT_EQ(err.message(), "test");
}

TEST(ErrorTest, Success) { EXPECT_TRUE(Error::Success().ok()); }

TEST(ErrorTest, ErrorUnused) {
  ASSERT_DEATH({ Error err("test"); }, "CHECK failure at");
}

auto IndirectError() -> Error { return Error("test"); }

TEST(ErrorTest, IndirectError) { EXPECT_FALSE(IndirectError().ok()); }

TEST(ErrorTest, ErrorOr) {
  ErrorOr<int> err(Error("test"));
  EXPECT_FALSE(err.ok());
  EXPECT_FALSE(err.error().ok());
  EXPECT_EQ(err.error().message(), "test");
}

TEST(ErrorTest, ErrorOrValue) { EXPECT_TRUE(ErrorOr<int>(0).ok()); }

TEST(ErrorTest, ErrorOrUnused) {
  ASSERT_DEATH({ ErrorOr<int> err(Error("test")); }, "CHECK failure at");
}

TEST(ErrorTest, ErrorOrUnusedVal) {
  ASSERT_DEATH({ ErrorOr<int> err(0); }, "CHECK failure at");
}

TEST(ErrorTest, ErrorOrSuccess) {
  ASSERT_DEATH({ ErrorOr<int> err(Error::Success()); }, "CHECK failure at");
}

auto IndirectErrorOrTest() -> ErrorOr<int> { return Error("test"); }

TEST(ErrorTest, IndirectErrorOr) { EXPECT_FALSE(IndirectErrorOrTest().ok()); }

}  // namespace
}  // namespace Carbon::Testing
