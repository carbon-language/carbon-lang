// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/common/error_builders.h"

#include <gtest/gtest.h>

#include "executable_semantics/common/source_location.h"

namespace Carbon::Testing {
namespace {

TEST(ErrorBuildersTest, CompilationError) {
  Error err = CompilationErrorBuilder(SourceLocation("x", 1)) << "test";
  EXPECT_EQ(err.message(), "COMPILATION ERROR: x:1: test");
}

TEST(ErrorBuildersTest, ProgramError) {
  Error err = ProgramErrorBuilder(SourceLocation("x", 1)) << "test";
  EXPECT_EQ(err.message(), "PROGRAM ERROR: x:1: test");
}

TEST(ErrorBuildersTest, RuntimeError) {
  Error err = RuntimeErrorBuilder(SourceLocation("x", 1)) << "test";
  EXPECT_EQ(err.message(), "RUNTIME ERROR: x:1: test");
}

}  // namespace
}  // namespace Carbon::Testing
