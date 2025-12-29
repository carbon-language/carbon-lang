// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_BAZEL_WORKING_DIR_H_
#define CARBON_COMMON_BAZEL_WORKING_DIR_H_

#include <stdlib.h>

#include <filesystem>
#include <system_error>

#include "common/check.h"
#include "common/filesystem.h"

namespace Carbon {

// Change working directory to behave as if it is where `bazel run` was invoked.
//
// Accepts an optional `exe_path` argument that will be adjusted to continue to
// be valid after this adjustment.
//
// There is no reasonable recovery we can do if either we can't make the path
// absolute or we can't change directory. As a consequence, this aborts if
// either of those fail rather than propagating any error.
inline auto SetWorkingDirForBazelRun(std::filesystem::path exe_path = {})
    -> std::filesystem::path {
  char* build_working_dir = getenv("BUILD_WORKING_DIRECTORY");
  if (build_working_dir == nullptr) {
    return exe_path;
  }

  // Adjust `exe_path` before changing directory.
  if (!exe_path.empty()) {
    std::error_code err;
    exe_path = std::filesystem::absolute(exe_path, err);
    CARBON_CHECK(!err, "Unable to make an absolute path for `{0}`: {1}",
                 exe_path, err.message());
  }

  auto chdir_result = Filesystem::Cwd().Chdir(build_working_dir);
  CARBON_CHECK(chdir_result.ok(),
               "Unable to change working directory to `{0}`: {1}",
               build_working_dir, chdir_result.error());

  return exe_path;
}

}  // namespace Carbon

#endif  // CARBON_COMMON_BAZEL_WORKING_DIR_H_
