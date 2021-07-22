// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/common/error.h"

#include "gtest/gtest.h"

namespace Carbon {

TEST(ErrorTest, FatalUserError) {
  ASSERT_DEATH({ FatalUserError() << "test"; }, "ERROR: test\n");
}

}  // namespace Carbon
