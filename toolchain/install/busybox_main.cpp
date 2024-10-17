// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <unistd.h>

#include <cstdlib>
#include <filesystem>

#include "common/bazel_working_dir.h"
#include "common/error.h"
#include "common/exe_path.h"
#include "common/init_llvm.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LLVMDriver.h"
#include "toolchain/driver/driver.h"
#include "toolchain/install/install_paths.h"

// Defined in:
// https://github.com/llvm/llvm-project/blob/main/clang/tools/driver/driver.cpp
//
// While not in a header, this is the API used by llvm-driver.cpp for
// busyboxing.
//
// TODO: For busyboxing, consider pulling in the full set of main functions
// here, similar to:
// https://github.com/llvm/llvm-project/blob/main/llvm/tools/llvm-driver/llvm-driver.cpp
//
// NOLINTNEXTLINE(readability-identifier-naming)
auto clang_main(int Argc, char** Argv, const llvm::ToolContext& ToolContext)
    -> int;

auto main(int argc, char** argv) -> int {
  Carbon::InitLLVM init_llvm(argc, argv);

  if (argc < 2) {
    llvm::errs() << "error: missing busybox target\n";
    return EXIT_FAILURE;
  }

  auto fs = llvm::vfs::getRealFileSystem();

  // Resolve paths before calling SetWorkingDirForBazel.
  std::string exe_path = Carbon::FindExecutablePath(argv[0]);
  const auto install_paths = Carbon::InstallPaths::MakeExeRelative(exe_path);
  if (install_paths.error()) {
    llvm::errs() << "error: " << *install_paths.error();
    return EXIT_FAILURE;
  }

  Carbon::SetWorkingDirForBazel();

  llvm::StringRef busybox_target = argv[1];
  llvm::SmallVector<llvm::StringRef> args;
  args.reserve(argc - 2);

  if (busybox_target == "carbon") {
    // No special edits.
  } else if (busybox_target == "clang") {
    // When clang is called with a `-cc1` flags, directly call `clang_main` to
    // get `ExecuteCC1Tool` handling. Otherwise, use Carbon's driver.
    if (argc >= 3) {
      llvm::StringRef maybe_cc1 = argv[2];
      if (maybe_cc1 == "-cc1" || maybe_cc1 == "-cc1as" ||
          maybe_cc1 == "-cc1gen-reproducer") {
        // Replace the `clang` argument with the exe path, to avoid creating a
        // new arg array.
        argv[1] = argv[0];

        // We need the prepend arg because `argv[0]` is `carbon-busybox`.
        llvm::ToolContext tool_context = {
            .Path = argv[0], .PrependArg = "clang", .NeedsPrependArg = true};
        return clang_main(argc - 1, argv + 1, tool_context);
      }
    }
    // Bounce through the `clang` subcommand.
    args.append({"clang", "--"});
  } else {
    llvm::errs() << "error: unsupported busybox name: " << busybox_target
                 << "\n";
    return EXIT_FAILURE;
  }

  args.append(argv + 2, argv + argc);
  Carbon::Driver driver(*fs, &install_paths, llvm::outs(), llvm::errs());
  bool success = driver.RunCommand(args).success;
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
