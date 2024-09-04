// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/install/install_paths.h"

#include <memory>

#include "common/check.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace Carbon {

// The location within our Bazel output tree of the prefix_root.
static constexpr llvm::StringLiteral PrefixRoot =
    "carbon/toolchain/install/prefix_root/";

// Path within an install prefix for our marker of a valid install.
static constexpr llvm::StringLiteral MarkerPath =
    "lib/carbon/carbon_install.txt";

auto InstallPaths::MakeExeRelative(llvm::StringRef exe_path) -> InstallPaths {
  InstallPaths paths;

  // Map from the executable path from the executable path to an install
  // prefix path.
  if (!llvm::sys::fs::exists(exe_path)) {
    paths.SetError(llvm::Twine("No file at executable path: ") + exe_path);
    return paths;
  }
  paths = InstallPaths(exe_path);

  // TODO: Detect a Windows executable path and use custom logic to map to the
  // correct install prefix for that platform.

  // We assume an executable will be in a `bin` directory and this is a
  // FHS-like install prefix. We remove the filename and walk up to find the
  // expected install prefix.
  llvm::sys::path::remove_filename(paths.prefix_);
  llvm::sys::path::append(paths.prefix_, llvm::sys::path::Style::posix, "../");

  paths.CheckMarkerFile();
  return paths;
}

auto InstallPaths::MakeForBazelRunfiles(llvm::StringRef exe_path)
    -> InstallPaths {
  using bazel::tools::cpp::runfiles::Runfiles;
  std::string runtimes_error;
  std::unique_ptr<Runfiles> runfiles(
      Runfiles::Create(exe_path.str(), &runtimes_error));
  CARBON_CHECK(runfiles != nullptr)
      << "Failed to find runtimes tree: " << runtimes_error;

  std::string relative_marker_path = (PrefixRoot.str() + MarkerPath).str();
  std::string runtimes_marker_path = runfiles->Rlocation(relative_marker_path);

  // Start from the marker, remove that filename, and walk up to find the
  // install prefix.
  InstallPaths paths(runtimes_marker_path);
  llvm::sys::path::remove_filename(paths.prefix_);
  llvm::sys::path::append(paths.prefix_, llvm::sys::path::Style::posix,
                          "../../");

  paths.CheckMarkerFile();
  CARBON_CHECK(!paths.error()) << *paths.error();
  return paths;
}

auto InstallPaths::Make(llvm::StringRef install_prefix) -> InstallPaths {
  InstallPaths paths(install_prefix);
  paths.CheckMarkerFile();
  return paths;
}

auto InstallPaths::FindPreludeFiles() const
    -> ErrorOr<llvm::SmallVector<std::string>> {
  // This is structured to avoid a vector copy on success.
  ErrorOr<llvm::SmallVector<std::string>> result =
      llvm::SmallVector<std::string>();

  std::string dir = core_package();

  // Include <data>/core/prelude.carbon, which is the entry point into the
  // prelude.
  {
    llvm::SmallString<256> prelude_file(dir);
    llvm::sys::path::append(prelude_file, llvm::sys::path::Style::posix,
                            "prelude.carbon");
    result->push_back(prelude_file.str().str());
  }

  // Glob for <data>/core/prelude/**/*.carbon and add all the files we find.
  llvm::SmallString<256> prelude_dir(dir);
  llvm::sys::path::append(prelude_dir, llvm::sys::path::Style::posix,
                          "prelude");
  std::error_code ec;
  for (llvm::sys::fs::recursive_directory_iterator prelude_files_it(
           prelude_dir, ec, /*follow_symlinks=*/false);
       prelude_files_it != llvm::sys::fs::recursive_directory_iterator();
       prelude_files_it.increment(ec)) {
    if (ec) {
      result = ErrorBuilder() << "Could not find prelude: " << ec.message();
      return result;
    }

    auto prelude_file = prelude_files_it->path();
    if (llvm::sys::path::extension(prelude_file) == ".carbon") {
      result->push_back(prelude_file);
    }
  }

  return result;
}

auto InstallPaths::SetError(llvm::Twine message) -> void {
  // Use an empty prefix on error as that should use the working directory which
  // is the least likely problematic.
  prefix_ = "";
  error_ = {message.str()};
}

auto InstallPaths::CheckMarkerFile() -> void {
  llvm::SmallString<256> path(prefix_);
  llvm::sys::path::append(path, llvm::sys::path::Style::posix, MarkerPath);
  if (!llvm::sys::fs::exists(path)) {
    SetError(llvm::Twine("No install marker at path: ") + path);
  }
}

auto InstallPaths::driver() const -> std::string {
  llvm::SmallString<256> path(prefix_);
  // TODO: Adjust this to work equally well on Windows.
  llvm::sys::path::append(path, llvm::sys::path::Style::posix, "bin/carbon");
  return path.str().str();
}

auto InstallPaths::core_package() const -> std::string {
  llvm::SmallString<256> path(prefix_);
  // TODO: Adjust this to work equally well on Windows.
  llvm::sys::path::append(path, llvm::sys::path::Style::posix,
                          "lib/carbon/core");
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
