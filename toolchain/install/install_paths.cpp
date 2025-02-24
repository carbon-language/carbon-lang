// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/install/install_paths.h"

#include <memory>

#include "common/check.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
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
  llvm::sys::path::append(paths.prefix_, llvm::sys::path::Style::posix,
                          "../../");

  if (auto error = llvm::sys::fs::make_absolute(paths.prefix_)) {
    paths.SetError(error.message());
    return paths;
  }

  paths.CheckMarkerFile();
  return paths;
}

auto InstallPaths::MakeForBazelRunfiles(llvm::StringRef exe_path)
    -> InstallPaths {
  using bazel::tools::cpp::runfiles::Runfiles;
  std::string runtimes_error;
  std::unique_ptr<Runfiles> runfiles(
      Runfiles::Create(exe_path.str(), &runtimes_error));
  CARBON_CHECK(runfiles != nullptr, "Failed to find runtimes tree: {0}",
               runtimes_error);

  std::string relative_marker_path = (PrefixRoot.str() + MarkerPath).str();
  std::string runtimes_marker_path = runfiles->Rlocation(relative_marker_path);

  // Start from the marker, remove that filename, and walk up to find the
  // install prefix.
  InstallPaths paths(runtimes_marker_path);
  llvm::sys::path::remove_filename(paths.prefix_);
  llvm::sys::path::append(paths.prefix_, llvm::sys::path::Style::posix,
                          "../../");

  if (auto error = llvm::sys::fs::make_absolute(paths.prefix_)) {
    paths.SetError(error.message());
    return paths;
  }

  paths.CheckMarkerFile();
  CARBON_CHECK(!paths.error(), "{0}", *paths.error());
  return paths;
}

auto InstallPaths::Make(llvm::StringRef install_prefix) -> InstallPaths {
  InstallPaths paths(install_prefix);
  paths.CheckMarkerFile();
  return paths;
}

auto InstallPaths::ReadPreludeManifest() const
    -> ErrorOr<llvm::SmallVector<std::string>> {
  // This is structured to avoid a vector copy on success.
  ErrorOr<llvm::SmallVector<std::string>> result =
      llvm::SmallVector<std::string>();

  llvm::SmallString<256> manifest;
  llvm::sys::path::append(manifest, llvm::sys::path::Style::posix,
                          core_package(), "prelude_manifest.txt");

  auto fs = llvm::vfs::getRealFileSystem();
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file =
      fs->getBufferForFile(manifest);
  if (!file) {
    result = ErrorBuilder() << "Loading prelude manifest `" << manifest
                            << "`: " << file.getError().message();
    return result;
  }

  // The manifest should have one file per line.
  llvm::StringRef buffer = file.get()->getBuffer();
  while (true) {
    auto [token, remainder] = llvm::getToken(buffer, "\n");
    if (token.empty()) {
      break;
    }
    llvm::SmallString<256> path;
    llvm::sys::path::append(path, llvm::sys::path::Style::posix, core_package(),
                            token);
    result->push_back(path.str().str());
    buffer = remainder;
  }

  if (result->empty()) {
    result = ErrorBuilder() << "Prelude manifest `" << manifest << "` is empty";
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
  if (!llvm::sys::path::is_absolute(prefix_)) {
    SetError(llvm::Twine("Not an absolute path: ") + prefix_);
  }

  llvm::SmallString<256> path(prefix_);
  llvm::sys::path::append(path, llvm::sys::path::Style::posix, MarkerPath);
  if (!llvm::sys::fs::exists(path)) {
    SetError(llvm::Twine("No install marker at path: ") + path);
  }
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

auto InstallPaths::clang_path() const -> std::string {
  llvm::SmallString<256> path(prefix_);
  // TODO: Adjust this to work equally well on Windows.
  llvm::sys::path::append(path, llvm::sys::path::Style::posix,
                          "lib/carbon/llvm/bin/clang");
  return path.str().str();
}

auto InstallPaths::lld_path() const -> std::string {
  llvm::SmallString<256> path(prefix_);
  // TODO: Adjust this to work equally well on Windows.
  llvm::sys::path::append(path, llvm::sys::path::Style::posix,
                          "lib/carbon/llvm/bin/lld");
  return path.str().str();
}

auto InstallPaths::ld_lld_path() const -> std::string {
  llvm::SmallString<256> path(prefix_);
  // TODO: Adjust this to work equally well on Windows.
  llvm::sys::path::append(path, llvm::sys::path::Style::posix,
                          "lib/carbon/llvm/bin/ld.lld");
  return path.str().str();
}

auto InstallPaths::ld64_lld_path() const -> std::string {
  llvm::SmallString<256> path(prefix_);
  // TODO: Adjust this to work equally well on Windows.
  llvm::sys::path::append(path, llvm::sys::path::Style::posix,
                          "lib/carbon/llvm/bin/ld64.lld");
  return path.str().str();
}

}  // namespace Carbon
