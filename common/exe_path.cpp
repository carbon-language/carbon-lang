// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/exe_path.h"

#include <string>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"

namespace Carbon {

auto FindExecutablePath(const char* argv0) -> std::string {
  static int static_for_main_addr;
  // Note this returns the canonical path, dropping symlink information that we
  // might want. As a consequence, we use it as a last resort. However, it's
  // also helpful to use to ensure we found the correct tool through other
  // means.
  std::string exe_path =
      llvm::sys::fs::getMainExecutable(argv0, &static_for_main_addr);

  llvm::StringRef argv0_ref = argv0;

  char realpath_buf[PATH_MAX];

  // If `argv[0]` is path-like and points at the executable, use the form in
  // `argv[0]`.
  if (argv0_ref.contains('/') && realpath(argv0, realpath_buf) &&
      realpath_buf == exe_path) {
    return argv0_ref.str();
  }

  // If we can find `argv[0]` in `$PATH`, use the form from that.
  //
  // For example, `llvm-symbolizer` is subprocessed with `argv[0]` that uses
  // this path. If `LLVM_SYMBOLIZER_PATH` is set to Carbon, but
  // `llvm-symbolizer` in `$PATH` is a different binary, that can lead to
  // problems -- which is why we verify the match.
  if (llvm::ErrorOr<std::string> path = llvm::sys::findProgramByName(argv0_ref);
      path && realpath(path->c_str(), realpath_buf) &&
      realpath_buf == exe_path) {
    return std::move(*path);
  }

  // As a fallback, use exe_path.
  return exe_path;
}

}  // namespace Carbon
