// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/common/error.h"

#include "gtest/gtest.h"

namespace Carbon {
namespace {

TEST(ErrorTest, FatalUserError) {
  ASSERT_DEATH({ FatalUserError(ErrorLine::None) << "test"; }, "ERROR: test\n");
}

TEST(ErrorTest, FatalRuntimeError) {
  ASSERT_DEATH({ FatalRuntimeError(ErrorLine::None) << "test"; },
               "RUNTIME ERROR: test\n");
}

TEST(ErrorTest, FatalCompilationError) {
  ASSERT_DEATH({ FatalCompilationError(ErrorLine::None) << "test"; },
               "COMPILATION ERROR: test\n");
}

TEST(ErrorTest, FatalUserErrorLine) {
  ASSERT_DEATH({ FatalUserError(1) << "test"; }, "ERROR: 1: test\n");
}

auto NoReturnRequired() -> int { FatalUserError(ErrorLine::None) << "test"; }

TEST(ErrorTest, NoReturnRequired) {
  ASSERT_DEATH({ NoReturnRequired(); }, "ERROR: test\n");
}

}  // namespace
}  // namespace Carbon
