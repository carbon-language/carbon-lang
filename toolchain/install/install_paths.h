// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_INSTALL_INSTALL_PATHS_H_
#define CARBON_TOOLCHAIN_INSTALL_INSTALL_PATHS_H_

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace Carbon {

// Locates the toolchain installation and provides paths to various components.
//
// The Carbon toolchain expects to be installed into some install prefix. For
// example, this is expected to be similar to the CMake install prefix:
//
// - `C:/Program Files/Carbon` or similar on Windows.
// - `/usr` or `/usr/local` on Linux and most BSDs.
// - `/opt/homebrew` or similar on macOS with Homebrew.
// - `bazel-bin/some/bazel/target.runfiles/_main/toolchain/install/prefix_root`
//   for unit tests and just-built binaries during development.
//
// See https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html
// for more details. While we don't build the toolchain with CMake, we expect
// our installation to behave in a similar and compatible way.
//
// There are multiple ways of locating an install's prefix for different
// situations. For command line tools distributed as part of the install, their
// own executable path is used to locate the rest of the install. We also
// support locating an install through Bazel's runfiles tree or through an
// explicit path for other use cases. When locating an install, we verify it by
// looking for the `carbon_install.txt` marker file at a specific location
// below. The install paths object retains any error information so that the
// driver can diagnose errors and report them as needed, but continue to
// function minimally. No methods will crash even in an error state, they will
// just return based on an empty install prefix.
//
// Within this prefix, we expect a hierarchy on Unix-y platforms:
//
// - `prefix_root/bin/carbon` - the main CLI driver
// - `prefix_root/lib/carbon/carbon_install.txt` - a marker for the install
// - `prefix_root/lib/carbon/...` - private data & binaries
//
// This is loosely based on the FHS (Filesystem Hierarchy Standard).
//
// An instance of this class provides methods that query for specific paths
// within the install. Note that we want to abstract away any platform
// differences in the installation layout, and so while there are some broad
// paths available here, like the `prefix` method, those should primarily be
// used for logging or debugging. When a specific part of the install is needed,
// a dedicated accessor should be added that computes the path for that
// component.
//
// Path accessor methods on the class return `llvm::StringRef` for any paths
// that are stored in the class, and a `std::string` for any that are computed
// on demand.
//
// TODO: Need to check the installation structure of LLVM on Windows and figure
// out what Carbon's should be within a Windows prefix and how much of the
// structure we can share with the Unix-y layout of the prefix.
class InstallPaths {
 public:
  // Provide the current executable's path to detect the correct installation
  // prefix path. This requires the toolchain to be in its installed layout.
  static auto MakeExeRelative(llvm::StringRef exe_path) -> InstallPaths;

  // Provide the current executable's path, and use that to detect a Bazel or
  // Bazel-compatible runfiles install prefix path. This should only be used
  // where it is reasonable to rely on this rather than a fixed install location
  // such as for internal development purposes or other Bazel users of the
  // Carbon library.
  static auto MakeForBazelRunfiles(llvm::StringRef exe_path) -> InstallPaths;

  // Provide an explicit install paths prefix. This is useful for testing or for
  // using Carbon in an environment with an unusual path to the installed files.
  static auto Make(llvm::StringRef install_prefix) -> InstallPaths;

  // Check for an error detecting the install paths correctly.
  //
  // An empty return means no errors encountered and the paths should work
  // correctly.
  //
  // A non-empty return means there was an error, and details of the error are
  // in the `StringRef` for inclusion in any user report.
  auto error() const -> std::optional<llvm::StringRef> { return error_; };

  // The computed installation prefix. This should correspond to the
  // `prefix_root` directory in Bazel's output, or to some prefix the toolchain
  // is installed into on a system such as `/usr/local` or `/home/$USER`.
  //
  // In the event of an error, this will be the empty string.
  auto prefix() const -> llvm::StringRef { return prefix_; }

  auto driver() const -> std::string;
  auto llvm_install_bin() const -> std::string;

 private:
  InstallPaths() : error_("No prefix provided!") {}
  explicit InstallPaths(llvm::StringRef prefix) : prefix_(prefix) {}

  auto SetError(llvm::Twine message) -> void;
  auto CheckMarkerFile() -> void;

  llvm::SmallString<256> prefix_;
  std::optional<std::string> error_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_INSTALL_INSTALL_PATHS_H_
