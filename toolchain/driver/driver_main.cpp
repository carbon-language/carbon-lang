// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdlib>

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/driver/driver.h"

auto main(int argc, char** argv) -> int {
  if (argc < 1) {
    return EXIT_FAILURE;
  }

  llvm::SmallVector<llvm::StringRef, 16> args(argv + 1, argv + argc);
  Carbon::Driver driver;
  bool success = driver.RunFullCommand(args);
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
