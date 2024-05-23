// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/install/install_paths.h"

#include "common/check.h"
#include "common/vlog.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace Carbon {

// The suffix added to an executable name in Bazel's output to find its
// runfiles tree.
static constexpr llvm::StringLiteral RunfilesSuffix = ".runfiles/";

// The workspace name to get to the root of our Bazel tree.
static constexpr llvm::StringLiteral Workspace = "_main/";

// The location within our Bazel output tree of the prefix_root.
static constexpr llvm::StringLiteral PrefixRoot =
    "toolchain/install/prefix_root/";

InstallPaths::InstallPaths(llvm::StringRef exe_path,
                           llvm::raw_ostream* vlog_stream)
    : vlog_stream_(vlog_stream) {
  // Map from the executable path from the executable path to an install prefix
  // path.

  // TODO: Detect a Windows executable path and use custom logic to map to the
  // correct install prefix for that platform.

  // First, see if the binary is in a `bin` directory. If so, remove it
  // assuming this is a FHS-like install prefix. We start with this pattern both
  // because we expect it to be the "production" pattern and so useful to
  // involve a minimal number of filesystem accesses, and because even in
  // testing or development builds, when we _can_ use this pattern, we prefer it
  // to match the behavior of genuine installations as closely as possible.
  prefix_.assign(exe_path);
  llvm::sys::path::remove_filename(prefix_);
  if (auto rbegin_it = llvm::sys::path::rbegin(prefix_),
      rend_it = llvm::sys::path::rend(prefix_);
      rbegin_it != rend_it) {
    // Step over a trailing `.` signifying the current directory. Whether this
    // occurs for paths is ambiguous between the documentation and the observed
    // behavior, so this code works to be resilient.
    if (*rbegin_it == ".") {
      ++rbegin_it;
    }
    if (rbegin_it != rend_it && *rbegin_it == "bin") {
      // Rather than trying to remove the `bin` directory, walk upwards by
      // appending `..`. We also assume POSIX style at this point.
      llvm::sys::path::append(prefix_, llvm::sys::path::Style::posix, "../");
      return;
    }
  }

  // Next check for a Bazel runfiles directory relative to the binary name. This
  // is used for `genrule` style execution within Bazel and for direct execution
  // of tests or other just-built binaries.
  prefix_.assign(exe_path);
  prefix_.append(RunfilesSuffix);
  if (llvm::sys::fs::is_directory(prefix_)) {
    llvm::sys::path::append(prefix_, llvm::sys::path::Style::posix, Workspace,
                            PrefixRoot);
    CARBON_CHECK(llvm::sys::fs::is_directory(prefix_))
        << "Found runfiles tree but not an install prefix directory!";
    return;
  }

  // Lastly, check for a test runtimes tree environment variable. This will
  // match when running inside a Bazel test's runfiles tree. We check this last
  // of all in part because that allows testing of all of the prior patterns.
  prefix_.assign(llvm::StringRef(getenv("TEST_SRCDIR")));
  if (llvm::sys::fs::is_directory(prefix_)) {
    llvm::sys::path::append(prefix_, llvm::sys::path::Style::posix, Workspace,
                            PrefixRoot);
    CARBON_CHECK(llvm::sys::fs::is_directory(prefix_))
        << "Found runfiles tree but not an install prefix directory!";
    return;
  }

  // Otherwise, we use whatever path remained from removing the filename of the
  // executable path. This isn't a great fallback, but we don't have a lot of
  // choices.
  CARBON_VLOG()
      << "Failed to detect a recognized install path, falling back to: "
      << prefix_;
}

auto InstallPaths::driver() const -> std::string {
  llvm::SmallString<256> path(prefix_);
  // TODO: Adjust this to work equally well on Windows.
  llvm::sys::path::append(path, llvm::sys::path::Style::posix, "bin/carbon");
  return path.str().str();
}

auto InstallPaths::llvm_install_bin() const -> std::string {
  llvm::SmallString<256> path(prefix_);
  // TODO: Adjust this to work equally well on Windows.
  llvm::sys::path::append(path, llvm::sys::path::Style::posix,
                          "lib/carbon/llvm/bin/");
  return path.str().str();
}

}  // namespace Carbon
