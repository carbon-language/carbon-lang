// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/common/error.h"

#include <gtest/gtest.h>

namespace Carbon {
namespace {

TEST(ErrorTest, FatalProgramError) {
  ASSERT_DEATH({ FATAL_PROGRAM_ERROR_NO_LINE() << "test"; },
               "^PROGRAM ERROR: test\n");
}

TEST(ErrorTest, FatalRuntimeError) {
  ASSERT_DEATH({ FATAL_RUNTIME_ERROR_NO_LINE() << "test"; },
               "^RUNTIME ERROR: test\n");
}

TEST(ErrorTest, FatalCompilationError) {
  ASSERT_DEATH({ FATAL_COMPILATION_ERROR_NO_LINE() << "test"; },
               "^COMPILATION ERROR: test\n");
}

TEST(ErrorTest, FatalProgramErrorLine) {
  ASSERT_DEATH({ FATAL_PROGRAM_ERROR(1) << "test"; },
               "^PROGRAM ERROR: 1: test\n");
}

}  // namespace
}  // namespace Carbon
