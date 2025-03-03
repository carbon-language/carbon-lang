// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <unistd.h>

#include <cstdlib>

#include "common/bazel_working_dir.h"
#include "common/error.h"
#include "common/exe_path.h"
#include "common/init_llvm.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LLVMDriver.h"
#include "toolchain/driver/driver.h"
#include "toolchain/install/busybox_info.h"
#include "toolchain/install/install_paths.h"

namespace Carbon {

// The actual `main` implementation. Can return an exit code or an `Error`
// (which causes EXIT_FAILRUE).
static auto Main(int argc, char** argv) -> ErrorOr<int> {
  InitLLVM init_llvm(argc, argv);

  // Start by resolving any symlinks.
  CARBON_ASSIGN_OR_RETURN(auto busybox_info,
                          GetBusyboxInfo(FindExecutablePath(argv[0])));

  auto fs = llvm::vfs::getRealFileSystem();

  // Resolve paths before calling SetWorkingDirForBazel.
  std::string exe_path = busybox_info.bin_path.string();
  const auto install_paths = InstallPaths::MakeExeRelative(exe_path);
  if (install_paths.error()) {
    return Error(*install_paths.error());
  }

  SetWorkingDirForBazel();

  llvm::SmallVector<llvm::StringRef> args;
  args.reserve(argc + 1);
  if (busybox_info.mode) {
    // Map busybox modes to the relevant subcommands with any flags needed to
    // emulate the requested command. Typically, our busyboxed binaries redirect
    // to a specific subcommand with some flags set and then pass the remaining
    // busybox arguments as positional arguments to that subcommand.
    //
    // TODO: Add relevant flags to the `clang` subcommand and add `clang`-based
    // symlinks to this like `clang++`.
    auto subcommand_args =
        llvm::StringSwitch<llvm::SmallVector<llvm::StringRef>>(
            *busybox_info.mode)
            .Case("ld.lld", {"lld", "--platform=gnu", "--"})
            .Case("ld64.lld", {"lld", "--platform=darwin", "--"})
            .Default({*busybox_info.mode, "--"});
    args.append(subcommand_args);
  }
  args.append(argv + 1, argv + argc);

  Driver driver(fs, &install_paths, stdin, &llvm::outs(), &llvm::errs(),
                /*fuzzing=*/false, /*enable_leaking=*/true);
  bool success = driver.RunCommand(args).success;
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace Carbon

auto main(int argc, char** argv) -> int {
  auto result = Carbon::Main(argc, argv);
  if (result.ok()) {
    return *result;
  } else {
    llvm::errs() << "error: " << result.error() << "\n";
    return EXIT_FAILURE;
  }
}
