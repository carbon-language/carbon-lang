// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_INSTALL_INSTALL_PATHS_H_
#define CARBON_TOOLCHAIN_INSTALL_INSTALL_PATHS_H_

#include "common/error.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace Carbon {

// Locates the toolchain installation and provides paths to various components.
//
// The Carbon toolchain expects to be installed into some install prefix; see
// `prefix_` for details. When locating an install, we verify it with
// `CheckMarkerFile`. When errors occur, `SetError` makes `error()`
// available for diagnostics and clears the install prefix (leaving things
// minimally functional).
//
// The factory methods locate the install prefix based on their use-case:
//
//   - `MakeExeRelative` for command line tools in an install.
//   - `MakeForBazelRunfiles` for locating through Bazel's runfile tree.
//   - `Make` for an explicit path, for example in tests.
//
// An instance of this class provides methods that query for specific paths
// within the install. Note that we want to abstract away any platform
// differences in the installation layout. When a specific part of the install
// is needed, a dedicated accessor should be added that computes the path for
// that component.
//
// TODO: Need to check the installation structure of LLVM on Windows and figure
// out what Carbon's should be within a Windows prefix and how much of the
// structure we can share with the Unix-y layout of the prefix.
//
// TODO: InstallPaths is typically called from places using a VFS (both tests
// and the Driver), but does not use a VFS itself. It currently only supports
// using the real filesystem, but should probably support a VFS.
class InstallPaths {
 public:
  // Provide the current executable's path to detect the correct installation
  // prefix path. This assumes the toolchain to be in its installed layout.
  //
  // If detection fails, this reverts to using the current working directory as
  // the install prefix, and the error detected can be checked with `errors()`.
  static auto MakeExeRelative(llvm::StringRef exe_path) -> InstallPaths;

  // Provide the current executable's path, and use that to detect a Bazel or
  // Bazel-compatible runfiles install prefix path. This should only be used
  // where it is reasonable to rely on this rather than a fixed install location
  // such as for internal development purposes or other Bazel users of the
  // Carbon library.
  //
  // This method of construction also ensures the result is valid. If detection
  // fails for any reason, it will `CARBON_CHECK` fail with the error message.
  static auto MakeForBazelRunfiles(llvm::StringRef exe_path) -> InstallPaths;

  // Provide an explicit install paths prefix, which must be absolute. This is
  // useful for testing or for using Carbon in an environment with an unusual
  // path to the installed files.
  static auto Make(llvm::StringRef install_prefix) -> InstallPaths;

  // Returns the contents of the prelude manifest file. This is the list of
  // files that define the prelude, and will always be non-empty on success.
  auto ReadPreludeManifest() const -> ErrorOr<llvm::SmallVector<std::string>>;

  // Check for an error detecting the install paths correctly.
  //
  // A nullopt return means no errors encountered and the paths should work
  // correctly.
  //
  // A string return means there was an error, and details of the error are
  // in the `StringRef` for inclusion in any user report.
  [[nodiscard]] auto error() const -> std::optional<llvm::StringRef> {
    return error_;
  }

  // The directory containing the `Core` package. Computed on demand.
  auto core_package() const -> std::string;

  // The directory containing LLVM install binaries. Computed on demand.
  auto llvm_install_bin() const -> std::string;

  // The path to `clang`.
  auto clang_path() const -> std::string;

  // The path to `lld' and various aliases of `lld`.
  auto lld_path() const -> std::string;
  auto ld_lld_path() const -> std::string;
  auto ld64_lld_path() const -> std::string;

 private:
  friend class InstallPathsTestPeer;

  InstallPaths() { SetError("No prefix provided!"); }
  explicit InstallPaths(llvm::StringRef prefix) : prefix_(prefix) {}

  // Set an error message on the install paths and reset the prefix to empty,
  // which should use the current working directory.
  auto SetError(llvm::Twine message) -> void;

  // Check that the install paths have a marker file at
  // `prefix()/lib/carbon/carbon_install.txt". If not, calls `SetError` with the
  // relevant error message.
  auto CheckMarkerFile() -> void;

  // The computed installation prefix. This will be an absolute path. We keep an
  // absolute path for when the command line uses a relative path
  // (`./bin/carbon`) and the working directory changes after initialization
  // (for example, to Bazel's working directory). In the event of an error, this
  // will be the empty string.
  //
  // When run from bazel (for example, in unit tests or development binaries)
  // this will look like:
  // `bazel-bin/some/bazel/target.runfiles/_main/toolchain/install/prefix_root`
  //
  // When installed, it's expected to be similar to the CMake install prefix:
  //
  // - `C:/Program Files/Carbon` or similar on Windows.
  // - `/usr` or `/usr/local` on Linux and most BSDs.
  // - `/opt/homebrew` or similar on macOS with Homebrew.
  //
  // See https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html
  // for more details. While we don't build the toolchain with CMake, we expect
  // our installation to behave in a similar and compatible way.
  //
  // The hierarchy of files beneath the install prefix can be found in the
  // BUILD's `install_dirs`.
  llvm::SmallString<256> prefix_;

  std::optional<std::string> error_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_INSTALL_INSTALL_PATHS_H_
