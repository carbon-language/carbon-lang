// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <stdio.h>

#include "llvm/Support/InitLLVM.h"

auto main(int argc, char** argv) -> int {
  printf("Running main() from gtest_main.cpp\n");
  testing::InitGoogleTest(&argc, argv);
  llvm::setBugReportMsg(
      "Please report issues to "
      "https://github.com/carbon-language/carbon-lang/issues and include the "
      "crash backtrace.\n");
  llvm::InitLLVM init_llvm(argc, argv);
  return RUN_ALL_TESTS();
}
