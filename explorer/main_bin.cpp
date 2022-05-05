// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/main.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

auto main(int argc, char** argv) -> int {
  // This assumes execution from bazel, in runfiles.
  llvm::SmallString<256> prelude_path;
  if (std::error_code err = llvm::sys::fs::current_path(prelude_path)) {
    llvm::errs() << "Failed to get working directory: " << err.message();
    return 1;
  }
  llvm::sys::path::append(prelude_path, "explorer/data/prelude.carbon");

  // Behave as if the working directory is where `bazel run` was invoked.
  char* build_working_dir = getenv("BUILD_WORKING_DIRECTORY");
  if (build_working_dir != nullptr) {
    if (std::error_code err =
            llvm::sys::fs::set_current_path(build_working_dir)) {
      llvm::errs() << "Failed to set working directory: " << err.message();
      return 1;
    }
  }

  return Carbon::ExplorerMain(prelude_path, argc, argv);
}
