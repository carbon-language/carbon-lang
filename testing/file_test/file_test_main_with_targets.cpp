// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/init_llvm_with_targets.h"
#include "testing/file_test/file_test_base.h"

auto main(int argc, char** argv) -> int {
  Carbon::InitLLVMWithTargets init_llvm(argc, argv);
  return Carbon::Testing::FileTestMain(argc, argv);
}
