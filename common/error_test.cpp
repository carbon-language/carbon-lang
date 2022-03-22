// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/error.h"

#include <gtest/gtest.h>

namespace Carbon::Testing {
namespace {

TEST(ErrorTest, Error) {
  Error err("test");
  EXPECT_EQ(err.message(), "test");
}

TEST(ErrorTest, ErrorEmptyString) {
  ASSERT_DEATH({ Error err(""); }, "CHECK failure at");
}

auto IndirectError() -> Error { return Error("test"); }

TEST(ErrorTest, IndirectError) { EXPECT_EQ(IndirectError().message(), "test"); }

TEST(ErrorTest, ErrorOr) {
  ErrorOr<int> err(Error("test"));
  EXPECT_FALSE(err.ok());
  EXPECT_EQ(err.error().message(), "test");
}

TEST(ErrorTest, ErrorOrValue) { EXPECT_TRUE(ErrorOr<int>(0).ok()); }

auto IndirectErrorOrTest() -> ErrorOr<int> { return Error("test"); }

TEST(ErrorTest, IndirectErrorOr) { EXPECT_FALSE(IndirectErrorOrTest().ok()); }

struct Val {
  int val;
};

TEST(ErrorTest, ErrorOrArrowOp) {
  ErrorOr<Val> err({1});
  EXPECT_EQ(err->val, 1);
}

auto IndirectErrorOrSuccessTest() -> ErrorOr<Success> { return Success(); }

TEST(ErrorTest, IndirectErrorOrSuccess) {
  EXPECT_TRUE(IndirectErrorOrSuccessTest().ok());
}

}  // namespace
}  // namespace Carbon::Testing
