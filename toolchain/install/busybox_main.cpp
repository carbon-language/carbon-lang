// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <unistd.h>

#include <cstdlib>
#include <string>

#include "clang/Driver/Driver.h"
#include "common/bazel_working_dir.h"
#include "common/error.h"
#include "common/init_llvm.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LLVMDriver.h"
#include "toolchain/base/install_paths.h"
#include "toolchain/driver/driver.h"
#include "toolchain/install/busybox_info.h"

namespace Carbon {

// The actual `main` implementation. Can return an exit code or an `Error`
// (which causes EXIT_FAILRUE).
static auto Main(int argc, char** argv) -> ErrorOr<int> {
  InitLLVM init_llvm(argc, argv);

  // Start by resolving any symlinks.
  CARBON_ASSIGN_OR_RETURN(auto busybox_info, GetBusyboxInfo(argv[0]));

  std::filesystem::path exe_path = busybox_info.bin_path.string();
  exe_path = SetWorkingDirForBazelRun(exe_path);

  const auto install_paths = InstallPaths::MakeExeRelative(exe_path.native());
  if (install_paths.error()) {
    return Error(*install_paths.error());
  }

  // If `LLVM_SYMBOLIZER_PATH` is unset, sets it. Signals.cpp would do some more
  // path resolution which this overrides in favor of using the busybox itself
  // for symbolization.
  setenv(
      "LLVM_SYMBOLIZER_PATH",
      (install_paths.llvm_install_bin().native() + "llvm-symbolizer").c_str(),
      /*overwrite=*/0);

  auto fs = llvm::vfs::getRealFileSystem();

  llvm::SmallVector<const char*> raw_args;
  raw_args.append(argv + 1, argv + argc);

  // Expand any response files in the arguments.
  llvm::BumpPtrAllocator alloc;
  if (llvm::Error error = clang::driver::expandResponseFiles(
          raw_args, busybox_info.mode && *busybox_info.mode == "clang-cl",
          alloc, fs.get())) {
    return Error(llvm::toString(std::move(error)));
  }

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
            // The `clang` program name used configures the default for its
            // `--driver-mode` flag. The first of these is redundant with the
            // default, but we group it here for clarity.
            .Case("clang", {"clang", "--"})
            .Case("clang++", {"clang", "--", "--driver-mode=g++"})
            .Case("clang-cl", {"clang", "--", "--driver-mode=cl"})
            .Case("clang-cpp", {"clang", "--", "--driver-mode=cpp"})

            // LLD has platform-specific program names that we translate into
            // platform flags.
            .Case("ld.lld", {"lld", "--platform=gnu", "--"})
            .Case("ld64.lld", {"lld", "--platform=darwin", "--"})

    // We also support a number of LLVM tools with a trivial translation
    // to subcommands. If any of these end up needing more advanced
    // translation, that can be factored into the `.def` file to provide custom
    // expansion here.
#define CARBON_LLVM_TOOL(Id, Name, BinName, MainFn) \
  .Case(BinName, {"llvm", Name, "--"})
#include "toolchain/base/llvm_tools.def"

            .Default({*busybox_info.mode, "--"});

    // And now append the subcommand args.
    args.append(subcommand_args);
  }
  llvm::append_range(args, raw_args);

  // We also support a special command line syntax for passing flags to the base
  // Carbon driver as `-Xcarbon=--some-carbon-flag=some-value`. This is
  // important when build systems only allow appending custom user flags to
  // allow them to be used for driver.
  //
  // Extract any arguments of that form, remove the prefix, and insert them to
  // the argument list just before the first positional parameter or subcommand.
  // This let's them come after any other flags to the base driver and override
  // them if needed.
  llvm::SmallVector<llvm::StringRef> extra_driver_args;
  llvm::erase_if(args, [&extra_driver_args](llvm::StringRef arg) {
    if (arg.consume_front("-Xcarbon=")) {
      extra_driver_args.push_back(arg);
      return true;
    }
    return false;
  });
  if (!extra_driver_args.empty()) {
    auto* subcommand_it = llvm::find_if(args, [](llvm::StringRef arg) {
      // Flags start with `-`, unless it is the string `-` or `--`.
      return !arg.starts_with("-") || arg == "-" || arg == "--";
    });
    args.insert(subcommand_it, extra_driver_args.begin(),
                extra_driver_args.end());
  }

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
