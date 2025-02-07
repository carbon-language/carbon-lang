// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/error.h"

#include <gtest/gtest.h>

#include "common/error_test_helpers.h"
#include "common/raw_string_ostream.h"

namespace Carbon {
namespace {

using ::Carbon::Testing::IsError;
using ::Carbon::Testing::IsSuccess;
using ::testing::Eq;

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

  EXPECT_THAT(err, IsError("test"));
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

TEST(ErrorTest, ErrorOrReference) {
  Val val = {1};
  ErrorOr<Val&> maybe_val(val);
  EXPECT_EQ(maybe_val->val, 1);
}

auto IndirectErrorOrSuccessTest() -> ErrorOr<Success> { return Success(); }

TEST(ErrorTest, IndirectErrorOrSuccess) {
  EXPECT_TRUE(IndirectErrorOrSuccessTest().ok());
}

TEST(ErrorTest, ReturnIfErrorNoError) {
  auto result = []() -> ErrorOr<Success> {
    CARBON_RETURN_IF_ERROR(ErrorOr<Success>(Success()));
    CARBON_RETURN_IF_ERROR(ErrorOr<Success>(Success()));
    return Success();
  }();
  EXPECT_TRUE(result.ok());
}

TEST(ErrorTest, ReturnIfErrorHasError) {
  auto result = []() -> ErrorOr<Success> {
    CARBON_RETURN_IF_ERROR(ErrorOr<Success>(Success()));
    CARBON_RETURN_IF_ERROR(ErrorOr<Success>(Error("error")));
    return Success();
  }();
  EXPECT_THAT(result, IsError("error"));
}

TEST(ErrorTest, AssignOrReturnNoError) {
  auto result = []() -> ErrorOr<int> {
    CARBON_ASSIGN_OR_RETURN(int a, ErrorOr<int>(1));
    CARBON_ASSIGN_OR_RETURN(const int b, ErrorOr<int>(2));
    int c = 0;
    CARBON_ASSIGN_OR_RETURN(c, ErrorOr<int>(3));
    return a + b + c;
  }();
  EXPECT_THAT(result, IsSuccess(Eq(6)));
}

TEST(ErrorTest, AssignOrReturnHasDirectError) {
  auto result = []() -> ErrorOr<int> {
    CARBON_RETURN_IF_ERROR(ErrorOr<int>(Error("error")));
    return 0;
  }();
  EXPECT_THAT(result, IsError("error"));
}

TEST(ErrorTest, AssignOrReturnHasErrorInExpected) {
  auto result = []() -> ErrorOr<int> {
    CARBON_ASSIGN_OR_RETURN(int a, ErrorOr<int>(Error("error")));
    return a;
  }();
  EXPECT_THAT(result, IsError("error"));
}

TEST(ErrorTest, ErrorBuilderOperatorImplicitCast) {
  ErrorOr<int> result = ErrorBuilder() << "msg";
  EXPECT_THAT(result, IsError("msg"));
}

TEST(ErrorTest, StreamError) {
  Error result = ErrorBuilder("TestFunc") << "msg";
  RawStringOstream result_stream;
  result_stream << result;
  EXPECT_EQ(result_stream.TakeStr(), "TestFunc: msg");
}

}  // namespace
}  // namespace Carbon
