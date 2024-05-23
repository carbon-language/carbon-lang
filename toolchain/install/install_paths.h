// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_INSTALL_INSTALL_PATHS_H_
#define CARBON_TOOLCHAIN_INSTALL_INSTALL_PATHS_H_

#include "common/ostream.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"

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
// We detect the installation prefix by looking at the path of the executable.
// This supports development binaries like tests, as well as any installed
// user-facing binaries.
//
// Within this prefix, we expect a hierarchy on Unix-y platforms:
//
// - `prefix_root/bin/carbon` - the main CLI driver.
// - `prefix_root/lib/carbon` - private data & binaries used by the toolchain.
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
  // prefix path. Optionally provide a verbose logging stream where details and
  // any issues computing the paths will be written.
  explicit InstallPaths(llvm::StringRef exe_path,
                        llvm::raw_ostream* vlog_stream = nullptr);

  // The computed installation prefix. This should correspond to the
  // `prefix_root` directory in Bazel's output, or to some prefix the toolchain
  // is installed into on a system such as `/usr/local` or `/home/$USER`.
  auto prefix() const -> llvm::StringRef { return prefix_; }

  auto driver() const -> std::string;
  auto llvm_install_bin() const -> std::string;

 private:
  llvm::SmallString<256> prefix_;
  llvm::raw_ostream* vlog_stream_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_INSTALL_INSTALL_PATHS_H_
