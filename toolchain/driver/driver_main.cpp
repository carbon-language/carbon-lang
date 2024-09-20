// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdlib>

#include "common/bazel_working_dir.h"
#include "common/exe_path.h"
#include "common/init_llvm.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/driver/driver.h"
#include "toolchain/install/install_paths.h"

auto main(int argc, char** argv) -> int {
  Carbon::InitLLVM init_llvm(argc, argv);

  if (argc < 1) {
    return EXIT_FAILURE;
  }

  // Resolve paths before calling SetWorkingDirForBazel.
  std::string exe_path = Carbon::FindExecutablePath(argv[0]);
  const auto install_paths = Carbon::InstallPaths::MakeExeRelative(exe_path);
  if (install_paths.error()) {
    llvm::errs() << "error: " << *install_paths.error();
    return EXIT_FAILURE;
  }

  Carbon::SetWorkingDirForBazel();

  llvm::SmallVector<llvm::StringRef> args(argv + 1, argv + argc);
  auto fs = llvm::vfs::getRealFileSystem();

  Carbon::Driver driver(*fs, &install_paths, llvm::outs(), llvm::errs());
  bool success = driver.RunCommand(args).success;
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
