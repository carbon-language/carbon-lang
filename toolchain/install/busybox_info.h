// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_INSTALL_BUSYBOX_INFO_H_
#define CARBON_TOOLCHAIN_INSTALL_BUSYBOX_INFO_H_

#include <filesystem>
#include <iterator>
#include <optional>
#include <string>

#include "common/error.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon {

constexpr const char* Argv0OverrideEnv = "CARBON_ARGV0_OVERRIDE";

struct BusyboxInfo {
  // The path to `carbon-busybox`.
  std::filesystem::path bin_path;
  // The mode, such as `carbon` or `clang`.
  std::optional<std::string> mode;
};

// Returns the busybox information, given argv[0].
//
// Extracts the desired mode for the busybox from the initial command name.
//
// Checks if the path in argv0 is an executable in a valid Carbon install, or a
// symlink to such an executable, and sets `bin_path` to the path of
// `lib/carbon/carbon-busybox` within that install.
//
// If unable to locate a plausible busybox binary, returns an error instead.
inline auto GetBusyboxInfo(llvm::StringRef argv0) -> ErrorOr<BusyboxInfo> {
  // Check for an override of `argv[0]` from the environment and apply it.
  if (const char* argv0_override = getenv(Argv0OverrideEnv)) {
    argv0 = argv0_override;
  }

  BusyboxInfo info = {.bin_path = argv0.str(), .mode = std::nullopt};
  std::filesystem::path filename = info.bin_path.filename();
  // The mode is set to the initial filename used for `argv[0]`.
  if (filename != "carbon" && filename != "carbon-busybox") {
    info.mode = filename;
  }

  // Now search through any symlinks to locate the installed busybox binary.
  while (true) {
    filename = info.bin_path.filename();
    if (filename == "carbon-busybox") {
      return info;
    }

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
      auto lib_path = filename == "carbon"
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
  }
}

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_INSTALL_BUSYBOX_INFO_H_
