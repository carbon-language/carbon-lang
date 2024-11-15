// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_INSTALL_BUSYBOX_INFO_H_
#define CARBON_TOOLCHAIN_INSTALL_BUSYBOX_INFO_H_

#include <filesystem>
#include <optional>
#include <string>

#include "common/error.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon {

struct BusyboxInfo {
  // The path to `carbon-busybox`.
  std::filesystem::path bin_path;
  // The mode, such as `carbon` or `clang`.
  std::optional<std::string> mode;
};

// Returns the busybox information, given argv[0]. This primarily handles
// resolving symlinks that point at the busybox.
inline auto GetBusyboxInfo(llvm::StringRef argv0) -> ErrorOr<BusyboxInfo> {
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
    // Do a path join, to handle relative symlinks.
    info.bin_path = info.bin_path.parent_path() / symlink_target;
  }
}

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_INSTALL_BUSYBOX_INFO_H_
