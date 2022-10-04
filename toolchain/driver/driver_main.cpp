// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdlib>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "toolchain/driver/driver.h"

auto main(int argc, char** argv) -> int {
  if (argc < 1) {
    return EXIT_FAILURE;
  }

  // Behave as if the working directory is where `bazel run` was invoked.
  char* build_working_dir = getenv("BUILD_WORKING_DIRECTORY");
  if (build_working_dir != nullptr) {
    if (std::error_code err =
            llvm::sys::fs::set_current_path(build_working_dir)) {
      llvm::errs() << "Failed to set working directory: " << err.message();
      return EXIT_FAILURE;
    }
  }

  llvm::SmallVector<llvm::StringRef, 16> args(argv + 1, argv + argc);
  Carbon::Driver driver;
  bool success = driver.RunFullCommand(args);
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
