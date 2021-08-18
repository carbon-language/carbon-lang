// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/common/ptr.h"

#include "gtest/gtest.h"

#include <iostream>

namespace Carbon {
namespace {

TEST(PtrTest, FatalProgramError) {
  Ptr<int> x;
  std::cerr << "Should not work: " << *x;
}

}  // namespace
}  // namespace Carbon
