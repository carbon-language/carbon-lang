// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_BAZEL_WORKING_DIR_H_
#define CARBON_COMMON_BAZEL_WORKING_DIR_H_

#include "llvm/Support/FileSystem.h"

namespace Carbon {

// Behave as if the working directory is where `bazel run` was invoked.
// This should only be used in development binaries, not release.
inline auto SetWorkingDirForBazel() -> bool {
  char* build_working_dir = getenv("BUILD_WORKING_DIRECTORY");
  if (build_working_dir == nullptr) {
    return true;
  }

  if (std::error_code err =
          llvm::sys::fs::set_current_path(build_working_dir)) {
    llvm::errs() << "Failed to set working directory: " << err.message();
    return false;
  }

  return true;
}

}  // namespace Carbon

#endif  // CARBON_COMMON_BAZEL_WORKING_DIR_H_
