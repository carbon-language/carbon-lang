// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/bazel_working_dir.h"
#include "explorer/main.h"

auto main(int argc, char** argv) -> int {
  Carbon::SetWorkingDirForBazel();

  static int static_for_main_addr;
  return Carbon::ExplorerMain(
      argc, argv, static_cast<void*>(&static_for_main_addr),
      // This assumes execution from `bazel-bin/explorer`, either directly or
      // with `bazel run`.
      "explorer.runfiles/carbon/explorer/data/prelude.carbon");
}
