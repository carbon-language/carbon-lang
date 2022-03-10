// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/common/error.h"

#include <gtest/gtest.h>

#include "llvm/Support/Error.h"

namespace Carbon::Testing {
namespace {

using ::testing::Eq;

auto MakeError(std::string_view message) -> llvm::Error {
  return llvm::make_error<llvm::StringError>(message,
                                             llvm::inconvertibleErrorCode());
}

auto MakeSuccess() -> llvm::Error { return llvm::Error::success(); }

auto MakeInt(int value) -> llvm::Expected<int> { return value; }

auto MakeFailedInt(std::string_view message) -> llvm::Expected<int> {
  return MakeError(message);
}

TEST(ErrorTest, FatalProgramError) {
  EXPECT_EQ(llvm::toString(FATAL_PROGRAM_ERROR_NO_LINE() << "test"),
            "PROGRAM ERROR: test");
}

TEST(ErrorTest, FatalRuntimeError) {
  EXPECT_EQ(llvm::toString(FATAL_RUNTIME_ERROR_NO_LINE() << "test"),
            "RUNTIME ERROR: test");
}

TEST(ErrorTest, FatalCompilationError) {
  EXPECT_EQ(llvm::toString(FATAL_COMPILATION_ERROR_NO_LINE() << "test"),
            "COMPILATION ERROR: test");
}

TEST(ErrorTest, FatalProgramErrorLine) {
  EXPECT_EQ(llvm::toString(FATAL_PROGRAM_ERROR(1) << "test"),
            "PROGRAM ERROR: 1: test");
}

TEST(ErrorTest, ReturnIfErrorNoError) {
  auto result = []() -> llvm::Error {
    RETURN_IF_ERROR(MakeSuccess());
    RETURN_IF_ERROR(MakeSuccess());
    return llvm::Error::success();
  }();
  EXPECT_FALSE(result);
}

TEST(ErrorTest, ReturnIfErrorHasError) {
  auto result = []() -> llvm::Error {
    RETURN_IF_ERROR(MakeSuccess());
    RETURN_IF_ERROR(MakeError("error"));
    return llvm::Error::success();
  }();
  ASSERT_TRUE(!!result);
  EXPECT_EQ(toString(std::move(result)), "error");
}

TEST(ErrorTest, AssignOrReturnNoError) {
  auto result = []() -> llvm::Expected<int> {
    RETURN_IF_ERROR(MakeSuccess());
    ASSIGN_OR_RETURN(int a, MakeInt(1));
    ASSIGN_OR_RETURN(const int b, MakeInt(2));
    int c = 0;
    ASSIGN_OR_RETURN(c, MakeInt(3));
    return a + b + c;
  }();
  ASSERT_TRUE(!!result);
  EXPECT_EQ(6, *result);
}

TEST(ErrorTest, AssignOrReturnHasDirectError) {
  auto result = []() -> llvm::Expected<int> {
    RETURN_IF_ERROR(MakeError("error"));
    return 0;
  }();
  ASSERT_FALSE(result);
}

TEST(ErrorTest, AssignOrReturnHasErrorInExpected) {
  auto result = []() -> llvm::Expected<int> {
    ASSIGN_OR_RETURN(int a, MakeFailedInt("error"));
    return a;
  }();
  ASSERT_FALSE(result);
  EXPECT_EQ(toString(result.takeError()), "error");
}

}  // namespace
}  // namespace Carbon::Testing
