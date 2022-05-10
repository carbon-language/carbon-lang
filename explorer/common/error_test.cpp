// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/common/error.h"

#include <gtest/gtest.h>

namespace Carbon::Testing {
namespace {

auto MakeSuccess() -> ErrorOr<Success> { return Success(); }

auto MakeError(std::string_view message) -> ErrorOr<Success> {
  return Error(message);
}

auto MakeInt(int value) -> ErrorOr<int> { return value; }

auto MakeFailedInt(std::string_view message) -> ErrorOr<int> {
  return Error(message);
}

auto ErrorToString(const Error& e) -> std::string { return e.message(); }

template <typename V>
auto ErrorToString(const ErrorOr<V>& e) -> std::string {
  return e.error().message();
}

TEST(ErrorTest, FatalProgramError) {
  EXPECT_EQ(ErrorToString(FATAL_PROGRAM_ERROR_NO_LINE() << "test"),
            "PROGRAM ERROR: test");
}

TEST(ErrorTest, FatalRuntimeError) {
  EXPECT_EQ(ErrorToString(FATAL_RUNTIME_ERROR_NO_LINE() << "test"),
            "RUNTIME ERROR: test");
}

TEST(ErrorTest, FatalCompilationError) {
  EXPECT_EQ(ErrorToString(FATAL_COMPILATION_ERROR_NO_LINE() << "test"),
            "COMPILATION ERROR: test");
}

TEST(ErrorTest, FatalProgramErrorLine) {
  EXPECT_EQ(ErrorToString(FATAL_PROGRAM_ERROR(1) << "test"),
            "PROGRAM ERROR: 1: test");
}

TEST(ErrorTest, ReturnIfErrorNoError) {
  auto result = []() -> ErrorOr<Success> {
    RETURN_IF_ERROR(MakeSuccess());
    RETURN_IF_ERROR(MakeSuccess());
    return Success();
  }();
  EXPECT_TRUE(result.ok());
}

TEST(ErrorTest, ReturnIfErrorHasError) {
  auto result = []() -> ErrorOr<Success> {
    RETURN_IF_ERROR(MakeSuccess());
    RETURN_IF_ERROR(MakeError("error"));
    return Success();
  }();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(ErrorToString(result), "error");
}

TEST(ErrorTest, AssignOrReturnNoError) {
  auto result = []() -> ErrorOr<int> {
    RETURN_IF_ERROR(MakeSuccess());
    ASSIGN_OR_RETURN(int a, MakeInt(1));
    ASSIGN_OR_RETURN(const int b, MakeInt(2));
    int c = 0;
    ASSIGN_OR_RETURN(c, MakeInt(3));
    return a + b + c;
  }();
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(6, *result);
}

TEST(ErrorTest, AssignOrReturnHasDirectError) {
  auto result = []() -> ErrorOr<int> {
    RETURN_IF_ERROR(MakeError("error"));
    return 0;
  }();
  ASSERT_FALSE(result.ok());
}

TEST(ErrorTest, AssignOrReturnHasErrorInExpected) {
  auto result = []() -> ErrorOr<int> {
    ASSIGN_OR_RETURN(int a, MakeFailedInt("error"));
    return a;
  }();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(ErrorToString(result), "error");
}

}  // namespace
}  // namespace Carbon::Testing
