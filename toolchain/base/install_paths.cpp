// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/install_paths.h"

#include <filesystem>
#include <memory>
#include <string>

#include "clang/Basic/Version.h"
#include "common/check.h"
#include "common/filesystem.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace Carbon {

// The location within our Bazel output tree of the install root.
static constexpr llvm::StringLiteral BazelRoot =
    "carbon/toolchain/install/prefix/lib/carbon/";

// Path within an install root for our marker of a valid install.
static constexpr llvm::StringLiteral MarkerPath = "carbon_install.txt";

auto InstallPaths::MakeExeRelative(llvm::StringRef exe_path) -> InstallPaths {
  InstallPaths paths;

  // Double check the exe was present.
  auto exe_access_result = Filesystem::Cwd().Access(exe_path.str());
  if (!exe_access_result.ok()) {
    paths.SetError(llvm::Twine("Failed to test for access executable: ") +
                   exe_access_result.error().ToString());
    return paths;
  } else if (!*exe_access_result) {
    paths.SetError(llvm::Twine("Unable to access executable: ") + exe_path);
    return paths;
  }

  return MakeFromFile(exe_path.str());
}

auto InstallPaths::MakeForBazelRunfiles(llvm::StringRef exe_path)
    -> InstallPaths {
  using bazel::tools::cpp::runfiles::Runfiles;
  std::string runtimes_error;
  std::unique_ptr<Runfiles> runfiles(
      Runfiles::Create(exe_path.str(), &runtimes_error));
  CARBON_CHECK(runfiles != nullptr, "Failed to find runtimes tree: {0}",
               runtimes_error);

  std::string relative_marker_path = (BazelRoot.str() + MarkerPath).str();
  std::filesystem::path runtimes_marker_path =
      runfiles->Rlocation(relative_marker_path);

  // Directly use the marker file's path.
  return MakeFromFile(std::move(runtimes_marker_path));
}

auto InstallPaths::Make(llvm::StringRef install_root) -> InstallPaths {
  InstallPaths paths(install_root.str());
  auto open_result = Filesystem::Cwd().OpenDir(paths.root_);
  if (!open_result.ok()) {
    paths.SetError(open_result.error().ToString());
  } else {
    paths.root_dir_ = *std::move(open_result);
    paths.CheckMarkerFile();
  }
  return paths;
}

auto InstallPaths::ReadPreludeManifest() const
    -> ErrorOr<llvm::SmallVector<std::string>> {
  return ReadManifest(core_package(), "prelude_manifest.txt");
}

auto InstallPaths::ReadClangHeadersManifest() const
    -> ErrorOr<llvm::SmallVector<std::string>> {
  // TODO: This is the only place where we read from outside of the install
  // root. Consider whether this manifest should be within the install or
  // consider moving the code to access it to be separate and specific to the
  // infrastructure needing it.
  return ReadManifest(root_ / "../../..", "clang_headers_manifest.txt");
}

auto InstallPaths::ReadManifest(std::filesystem::path manifest_path,
                                std::filesystem::path manifest_file) const
    -> ErrorOr<llvm::SmallVector<std::string>> {
  // This is structured to avoid a vector copy on success.
  ErrorOr<llvm::SmallVector<std::string>> result =
      llvm::SmallVector<std::string>();

  // TODO: It would be nice to adjust the manifests to be within the install
  // root and use that open directory to access the manifest. Also to update
  // callers to be able to use the relative paths via an open directory rather
  // than having to form absolute paths for all the entries.
  auto read_result =
      Filesystem::Cwd().ReadFileToString(manifest_path / manifest_file);
  if (!read_result.ok()) {
    result = ErrorBuilder()
             << "Loading manifest `" << (manifest_path / manifest_file)
             << "`: " << read_result.error();
    return result;
  }

  // The manifest should have one file per line.
  llvm::StringRef buffer = *read_result;
  while (true) {
    auto [token, remainder] = llvm::getToken(buffer, "\n");
    if (token.empty()) {
      break;
    }
    result->push_back((manifest_path / std::string_view(token)).native());
    buffer = remainder;
  }

  if (result->empty()) {
    result = ErrorBuilder()
             << "Manifest `" << (manifest_path / manifest_file) << "` is empty";
  }
  return result;
}

auto InstallPaths::MakeFromFile(std::filesystem::path file_path)
    -> InstallPaths {
  // TODO: Add any custom logic needed to detect the correct install root on
  // Windows once we have support for that platform.
  //
  // We assume the provided path is either the marker file itself or the busybox
  // executable that is adjacent to the marker file. That means the root is just
  // the directory of this path.
  InstallPaths paths(std::move(file_path).remove_filename());

  auto open_result = Filesystem::Cwd().OpenDir(paths.root_);
  if (!open_result.ok()) {
    paths.SetError(open_result.error().ToString());
    return paths;
  }

  paths.root_dir_ = *std::move(open_result);
  paths.CheckMarkerFile();
  return paths;
}

auto InstallPaths::SetError(llvm::Twine message) -> void {
  // Use an empty root on error as that should use the working directory which
  // is the least likely problematic.
  root_ = "";
  root_dir_ = Filesystem::Dir();
  error_ = {message.str()};
}

auto InstallPaths::CheckMarkerFile() -> void {
  auto access_result = root_dir_.Access(MarkerPath.str());
  if (!access_result.ok()) {
    SetError(access_result.error().ToString());
    return;
  }
  if (!*access_result) {
    SetError(llvm::Twine("No install marker at path: ") +
             (root_ / std::string_view(MarkerPath)).native());
    return;
  }

  // Success!
}

auto InstallPaths::core_package() const -> std::filesystem::path {
  // TODO: Adjust this to work equally well on Windows.
  return root_ / "core";
}

auto InstallPaths::llvm_install_bin() const -> std::filesystem::path {
  // TODO: Adjust this to work equally well on Windows.
  return root_ / "llvm/bin";
}

auto InstallPaths::clang_path() const -> std::filesystem::path {
  // TODO: Adjust this to work equally well on Windows.
  return llvm_install_bin() / "clang";
}

auto InstallPaths::lld_path() const -> std::filesystem::path {
  // TODO: Adjust this to work equally well on Windows.
  return llvm_install_bin() / "lld";
}

auto InstallPaths::ld_lld_path() const -> std::filesystem::path {
  // TODO: Adjust this to work equally well on Windows.
  return llvm_install_bin() / "ld.lld";
}

auto InstallPaths::ld64_lld_path() const -> std::filesystem::path {
  // TODO: Adjust this to work equally well on Windows.
  return llvm_install_bin() / "ld64.lld";
}

auto InstallPaths::llvm_tool_path(LLVMTool tool) const
    -> std::filesystem::path {
  // TODO: Adjust this to work equally well on Windows.
  return llvm_install_bin() / std::string_view(tool.bin_name());
}

auto InstallPaths::clang_resource_path() const -> std::filesystem::path {
  // TODO: Adjust this to work equally well on Windows.
  return root_ / "llvm/lib/clang/" CLANG_VERSION_MAJOR_STRING;
}

auto InstallPaths::runtimes_root() const -> std::filesystem::path {
  // TODO: Adjust this to work equally well on Windows.
  return root_ / "runtimes";
}

auto InstallPaths::libunwind_path() const -> std::filesystem::path {
  // TODO: Adjust this to work equally well on Windows.
  return runtimes_root() / "libunwind";
}

auto InstallPaths::libcxx_path() const -> std::filesystem::path {
  // TODO: Adjust this to work equally well on Windows.
  return runtimes_root() / "libcxx";
}

auto InstallPaths::libcxxabi_path() const -> std::filesystem::path {
  // TODO: Adjust this to work equally well on Windows.
  return runtimes_root() / "libcxxabi";
}

auto InstallPaths::libc_path() const -> std::filesystem::path {
  // TODO: Adjust this to work equally well on Windows.
  return runtimes_root() / "libc";
}

auto InstallPaths::digest_path() const -> std::filesystem::path {
  // TODO: Adjust this to work equally well on Windows.
  return root_ / "install_digest.txt";
}

}  // namespace Carbon
