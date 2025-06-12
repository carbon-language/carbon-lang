// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/install/busybox_info.h"

#include <iterator>

#include "common/exe_path.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon {

// The mode is set to the initial filename used for `argv[0]`.
static auto GetMode(const std::filesystem::path& argv0)
    -> std::optional<std::string> {
  std::string filename = argv0.filename();
  if (filename != "carbon" && filename != "carbon-busybox") {
    return filename;
  }
  return std::nullopt;
}

auto GetBusyboxInfo(const char* argv0) -> ErrorOr<BusyboxInfo> {
  // Need storage due to `unsetenv` affecting `getenv` lifetime; using `path`
  // for `GetMode`.
  std::filesystem::path argv0_path = argv0;

  // Check for an override of `argv[0]` from the environment and apply it.
  if (const char* argv0_override = getenv(Argv0OverrideEnv)) {
    argv0_path = argv0_override;
    unsetenv(Argv0OverrideEnv);
  }

  BusyboxInfo info = {.bin_path = FindExecutablePath(argv0_path.c_str()),
                      .mode = GetMode(argv0_path)};

  if (info.bin_path.filename() == "carbon-busybox") {
    // Check for bazel structure. For example, this makes work:
    //   /bin/sh -c "exec -a carbon ./bazel-bin/toolchain/carbon"
    //   /bin/sh -c "exec -a llvm-symbolizer ./bazel-bin/toolchain/carbon"
    if (std::filesystem::exists(info.bin_path.string() + ".runfiles")) {
      std::string prefix_root = info.bin_path.parent_path().string() +
                                "/prefix_root/lib/carbon/carbon-busybox";
      if (std::filesystem::exists(prefix_root)) {
        info.bin_path = prefix_root;
      }
    }
    return info;
  }

  // Now search through any symlinks to locate the installed busybox binary.
  while (true) {
    // If we've not already reached the busybox, look for it relative to the
    // current binary path. This can help more immediately locate an
    // installation tree, and avoids walking through a final layer of symlinks
    // which may point to content-addressed storage or other parts of a build
    // output tree.
    //
    // We break this into two cases we need to handle:
    // - Carbon's CLI will be: `<prefix>/bin/carbon`
    // - Other tools will be: `<prefix>/lib/carbon/<group>/bin/<tool>`
    //
    // We also check that the current path is within a `bin` directory to
    // provide best-effort checking for accidentally walking up from symlinks
    // that aren't within an installation-shaped tree.
    auto parent_path = info.bin_path.parent_path();
    // Strip any `.` path components at the end to simplify processing.
    while (parent_path.filename() == ".") {
      parent_path = parent_path.parent_path();
    }
    if (parent_path.filename() == "bin") {
      auto lib_path = info.bin_path.filename() == "carbon"
                          ? parent_path / ".." / "lib" / "carbon"
                          : parent_path / ".." / "..";
      auto busybox_path = lib_path / "carbon-busybox";
      std::error_code ec;
      if (std::filesystem::exists(busybox_path, ec)) {
        info.bin_path = busybox_path;
        return info;
      }
    }

    // Try to walk through another layer of symlinks and see if we can find the
    // installation there or are linked directly to the busybox.
    std::error_code ec;
    auto symlink_target = std::filesystem::read_symlink(info.bin_path, ec);
    if (ec) {
      return ErrorBuilder()
             << "expected carbon-busybox symlink at `" << info.bin_path << "`";
    }

    // Do a path join, to handle relative symlinks.
    info.bin_path = parent_path / symlink_target;
    if (info.bin_path.filename() == "carbon-busybox") {
      return info;
    }
  }
}

}  // namespace Carbon
