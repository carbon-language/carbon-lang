// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdlib>

#include "common/bazel_working_dir.h"
#include "common/exe_path.h"
#include "common/init_llvm.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"
#include "toolchain/driver/driver.h"
#include "toolchain/install/install_paths.h"

auto main(int argc, char** argv) -> int {
  Carbon::InitLLVM init_llvm(argc, argv);

  if (argc < 1) {
    return EXIT_FAILURE;
  }

  std::string exe_path = Carbon::FindExecutablePath(argv[0]);

  Carbon::SetWorkingDirForBazel();

  // Printing to stderr should flush stdout. This is most noticeable when stderr
  // is piped to stdout.
  llvm::errs().tie(&llvm::outs());

  llvm::SmallVector<llvm::StringRef> args(argv + 1, argv + argc);
  auto fs = llvm::vfs::getRealFileSystem();

  const auto install_paths = Carbon::InstallPaths::MakeExeRelative(exe_path);

  // Construct the data directory relative to the executable location.
  // TODO: Will be removed when everything moves to the install_paths.
  llvm::SmallString<256> data_dir(llvm::sys::path::parent_path(exe_path));
  llvm::sys::path::append(data_dir, llvm::sys::path::Style::posix,
                          "carbon.runfiles/_main/");

  Carbon::Driver driver(*fs, &install_paths, data_dir, llvm::outs(),
                        llvm::errs());
  bool success = driver.RunCommand(args).success;
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
