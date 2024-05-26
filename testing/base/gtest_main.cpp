// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/base/gtest_main.h"

#include <gtest/gtest.h>

#include <string>

#include "common/check.h"
#include "common/exe_path.h"
#include "common/init_llvm.h"

static bool after_main = false;
static llvm::StringRef exe_path;

namespace Carbon::Testing {

auto TestExePath() -> llvm::StringRef {
  CARBON_CHECK(after_main)
      << "Must not query the executable path until after `main` is entered!";
  return exe_path;
}

}  // namespace Carbon::Testing

auto main(int argc, char** argv) -> int {
  std::string exe_path_storage = Carbon::FindExecutablePath(argv[0]);
  exe_path = exe_path_storage;
  after_main = true;

  Carbon::InitLLVM init_llvm(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
