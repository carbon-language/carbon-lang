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

namespace Carbon {

namespace {
struct BusyboxInfo {
  // The path to `carbon-busybox`.
  std::filesystem::path bin_path;
  // The mode, such as `carbon` or `clang`.
  std::optional<std::string> mode;
};
}  // namespace

// Returns the busybox information, given argv[0]. This primarily handles
// resolving symlinks that point at the busybox.
static auto GetBusyboxInfo(llvm::StringRef argv0) -> ErrorOr<BusyboxInfo> {
  BusyboxInfo info = BusyboxInfo{argv0.str(), std::nullopt};
  while (true) {
    std::string filename = info.bin_path.filename();
    if (filename == "carbon-busybox") {
      return info;
    }
    std::error_code ec;
    auto symlink_target = std::filesystem::read_symlink(info.bin_path, ec);
    if (ec) {
      return ErrorBuilder()
             << "expected carbon-busybox symlink at `" << info.bin_path << "`";
    }
    info.mode = filename;
    info.bin_path = symlink_target;
  }
}

// The actual `main` implementation. Can return an exit code or an `Error`
// (which causes EXIT_FAILRUE).
static auto Main(int argc, char** argv) -> ErrorOr<int> {
  Carbon::InitLLVM init_llvm(argc, argv);

  // Start by resolving any symlinks.
  CARBON_ASSIGN_OR_RETURN(auto busybox_info, Carbon::GetBusyboxInfo(argv[0]));

  auto fs = llvm::vfs::getRealFileSystem();

  // Resolve paths before calling SetWorkingDirForBazel.
  std::string exe_path =
      Carbon::FindExecutablePath(busybox_info.bin_path.string());
  const auto install_paths = Carbon::InstallPaths::MakeExeRelative(exe_path);
  if (install_paths.error()) {
    return Error(*install_paths.error());
  }

  Carbon::SetWorkingDirForBazel();

  llvm::SmallVector<llvm::StringRef> args;
  args.reserve(argc + 1);
  if (busybox_info.mode && busybox_info.mode != "carbon") {
    args.append({*busybox_info.mode, "--"});
  }
  args.append(argv + 1, argv + argc);

  Carbon::Driver driver(*fs, &install_paths, llvm::outs(), llvm::errs());
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
