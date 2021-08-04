// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/common/error.h"

#include "gtest/gtest.h"

namespace Carbon {
namespace {

TEST(ErrorTest, FatalUserError) {
  ASSERT_DEATH({ FATAL_RUNTIME_ERROR_NO_LINE() << "test"; }, "ERROR: test\n");
}

TEST(ErrorTest, FatalRuntimeError) {
  ASSERT_DEATH({ FATAL_RUNTIME_ERROR_NO_LINE() << "test"; },
               "RUNTIME ERROR: test\n");
}

TEST(ErrorTest, FatalCompilationError) {
  ASSERT_DEATH({ FATAL_COMPILATION_ERROR_NO_LINE() << "test"; },
               "COMPILATION ERROR: test\n");
}

TEST(ErrorTest, FatalInternalError) {
  ASSERT_DEATH({ FATAL_INTERNAL_ERROR_NO_LINE() << "test"; },
               "INTERNAL ERROR: test\n");
}

TEST(ErrorTest, FatalUserErrorLine) {
  ASSERT_DEATH({ FATAL_USER_ERROR(1) << "test"; }, "ERROR: 1: test\n");
}

auto NoReturnRequired() -> int { FATAL_USER_ERROR_NO_LINE() << "test"; }

TEST(ErrorTest, NoReturnRequired) {
  ASSERT_DEATH({ NoReturnRequired(); }, "ERROR: test\n");
}

}  // namespace
}  // namespace Carbon
