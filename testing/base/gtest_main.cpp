// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>

#include "common/init_llvm.h"
#include "testing/base/global_exe_path.h"

auto main(int argc, char** argv) -> int {
  // Initialize LLVM first, as that will also handle ensuring UTF-8 encoding.
  Carbon::InitLLVM init_llvm(argc, argv);

  Carbon::Testing::SetExePath(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
