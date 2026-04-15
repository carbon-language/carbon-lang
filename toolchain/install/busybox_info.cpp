// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/install/busybox_info.h"

#include <iterator>

#include "common/exe_path.h"
#include "common/filesystem.h"
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

// Try to walk up the path using `.parent_path()` if we can to avoid extra
// components to resolve. However, if the path is relative to the current
// working directory and we run out of parent components, walk up by appending
// `../` components instead.
static auto WalkUp(std::filesystem::path p) -> std::filesystem::path {
  // Remove `./` components.
  while (p.filename() == ".") {
    p = p.parent_path();
  }
  if (!p.is_absolute() && (p.empty() || p.filename() == "..")) {
    return p / "..";
  } else {
    return p.parent_path();
  }
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

  // Now search through any symlinks to locate the installed busybox binary.
  while (true) {
    if (info.bin_path.filename() == "carbon-busybox") {
      return info;
    }

    // If we've not already reached the busybox, look for it relative to the
    // current binary path. This can help more immediately locate an
    // installation tree, and avoids walking through a final layer of symlinks
    // which may point to content-addressed storage or other parts of a build
    // output tree.
    //
    // We break this into two cases we need to handle:
    // - An install using the Unix-style FHS layout: `<prefix>/bin/carbon`
    // - Tools within the Carbon install root: `<install>/<group>/bin/<tool>`
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
      // Note that we use a specialized approach to walking up rather than
      // always appending `../` components. While largely equivalent, this helps
      // keep paths shorter and avoids redundant work. We also don't expect to
      // need to respect _internally_ strange symlinking structures that would
      // need to use appended `../` components.
      auto lib_path = info.bin_path.filename() == "carbon"
                          ? WalkUp(std::move(parent_path)) / "lib" / "carbon"
                          : WalkUp(WalkUp(std::move(parent_path)));
      auto busybox_path = lib_path / "carbon-busybox";
      if (auto access = Filesystem::Cwd().Access(busybox_path);
          access.ok() && *access) {
        info.bin_path = busybox_path;
        return info;
      }
    }

    // Try to walk through another layer of symlinks and see if we can find the
    // installation there or are linked directly to the busybox.
    auto readlink = Filesystem::Cwd().Readlink(info.bin_path);
    if (!readlink.ok()) {
      return ErrorBuilder()
             << "expected carbon-busybox symlink at `" << info.bin_path << "`";
    }

    // Do a path join, to handle relative symlinks.
    info.bin_path = parent_path / *readlink;
  }
}

}  // namespace Carbon
