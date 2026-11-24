// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/exe_path.h"

#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

namespace Carbon {

// Returns true if a found path resolves to the actual executable path.
static auto RealPathMatches(llvm::StringRef found_path,
                            llvm::StringRef exe_path) -> bool {
  llvm::SmallString<256> resolved;
  if (llvm::sys::fs::real_path(found_path, resolved)) {
    return false;
  }
#ifdef _WIN32
  return resolved.equals_insensitive(exe_path);
#else
  return resolved == exe_path;
#endif
}

auto FindExecutablePath(const char* argv0) -> std::string {
  static int static_for_main_addr;
  // Note this returns the canonical path, dropping symlink information that we
  // might want. As a consequence, we use it as a last resort. However, it's
  // also helpful to use to ensure we found the correct tool through other
  // means.
  std::string exe_path =
      llvm::sys::fs::getMainExecutable(argv0, &static_for_main_addr);

  llvm::StringRef argv0_ref = argv0;

  // If `argv[0]` is path-like and points at the executable, use the form in
  // `argv[0]`.
  bool has_separator = llvm::any_of(
      argv0_ref, [](char c) { return llvm::sys::path::is_separator(c); });
  if (has_separator) {
    if (RealPathMatches(argv0_ref, exe_path)) {
      return argv0_ref.str();
    }
#ifdef _WIN32
    if (!argv0_ref.ends_with_insensitive(".exe") &&
        RealPathMatches((argv0_ref + ".exe").str(), exe_path)) {
      return argv0_ref.str();
    }
#endif
  }

  // If we can find `argv[0]` in `$PATH`, use the form from that.
  //
  // For example, `llvm-symbolizer` is subprocessed with `argv[0]` that uses
  // this path. If `LLVM_SYMBOLIZER_PATH` is set to Carbon, but
  // `llvm-symbolizer` in `$PATH` is a different binary, that can lead to
  // problems -- which is why we verify the match.
  if (llvm::ErrorOr<std::string> path = llvm::sys::findProgramByName(argv0_ref);
      path && RealPathMatches(*path, exe_path)) {
    return std::move(*path);
  }

  // As a fallback, use exe_path.
  return exe_path;
}

}  // namespace Carbon
