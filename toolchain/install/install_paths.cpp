// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/install/install_paths.h"

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

// The location within our Bazel output tree of the prefix_root.
static constexpr llvm::StringLiteral PrefixRoot =
    "carbon/toolchain/install/prefix_root/";

// Path within an install prefix for our marker of a valid install.
static constexpr llvm::StringLiteral MarkerPath =
    "lib/carbon/carbon_install.txt";

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

  std::string relative_marker_path = (PrefixRoot.str() + MarkerPath).str();
  std::filesystem::path runtimes_marker_path =
      runfiles->Rlocation(relative_marker_path);

  // Start from the marker, remove that filename, and walk up to find the
  // install prefix.
  return MakeFromFile(std::move(runtimes_marker_path));
}

auto InstallPaths::Make(llvm::StringRef install_prefix) -> InstallPaths {
  InstallPaths paths(install_prefix.str());
  auto open_result = Filesystem::Cwd().OpenDir(paths.prefix_);
  if (!open_result.ok()) {
    paths.SetError(open_result.error().ToString());
  } else {
    paths.prefix_dir_ = *std::move(open_result);
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
  return ReadManifest(prefix_ / "..", "clang_headers_manifest.txt");
}

auto InstallPaths::ReadManifest(std::filesystem::path manifest_path,
                                std::filesystem::path manifest_file) const
    -> ErrorOr<llvm::SmallVector<std::string>> {
  // This is structured to avoid a vector copy on success.
  ErrorOr<llvm::SmallVector<std::string>> result =
      llvm::SmallVector<std::string>();

  // TODO: It would be nice to adjust the manifests to be within the install
  // prefix and use that open directory to access the manifest. Also to update
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
  // TODO: Detect a Windows executable path and use custom logic to map to the
  // correct install prefix for that platform.
  //
  // We assume an executable will be in a `bin` directory and this is a
  // FHS-like install prefix. We remove the filename and walk up to find the
  // expected install prefix.
  std::error_code ec;
  InstallPaths paths(std::filesystem::absolute(
      std::move(file_path).remove_filename() / "../..", ec));
  if (ec) {
    paths.SetError(ec.message());
    return paths;
  }

  auto open_result = Filesystem::Cwd().OpenDir(paths.prefix_);
  if (!open_result.ok()) {
    paths.SetError(open_result.error().ToString());
    return paths;
  }

  paths.prefix_dir_ = *std::move(open_result);
  paths.CheckMarkerFile();
  return paths;
}

auto InstallPaths::SetError(llvm::Twine message) -> void {
  // Use an empty prefix on error as that should use the working directory which
  // is the least likely problematic.
  prefix_ = "";
  prefix_dir_ = Filesystem::Dir();
  error_ = {message.str()};
}

auto InstallPaths::CheckMarkerFile() -> void {
  if (!prefix_.is_absolute()) {
    SetError(llvm::Twine("Not an absolute path: ") + prefix_.native());
    return;
  }

  auto access_result = prefix_dir_.Access(MarkerPath.str());
  if (!access_result.ok()) {
    SetError(access_result.error().ToString());
    return;
  }
  if (!*access_result) {
    SetError(llvm::Twine("No install marker at path: ") +
             (prefix_ / std::string_view(MarkerPath)).native());
    return;
  }

  // Success!
}

auto InstallPaths::core_package() const -> std::filesystem::path {
  // TODO: Adjust this to work equally well on Windows.
  return prefix_ / "lib/carbon/core";
}

auto InstallPaths::llvm_install_bin() const -> std::filesystem::path {
  // TODO: Adjust this to work equally well on Windows.
  return prefix_ / "lib/carbon/llvm/bin/";
}

auto InstallPaths::clang_path() const -> std::filesystem::path {
  // TODO: Adjust this to work equally well on Windows.
  return prefix_ / "lib/carbon/llvm/bin/clang";
}

auto InstallPaths::lld_path() const -> std::filesystem::path {
  // TODO: Adjust this to work equally well on Windows.
  return prefix_ / "lib/carbon/llvm/bin/lld";
}

auto InstallPaths::ld_lld_path() const -> std::filesystem::path {
  // TODO: Adjust this to work equally well on Windows.
  return prefix_ / "lib/carbon/llvm/bin/ld.lld";
}

auto InstallPaths::ld64_lld_path() const -> std::filesystem::path {
  // TODO: Adjust this to work equally well on Windows.
  return prefix_ / "lib/carbon/llvm/bin/ld64.lld";
}

auto InstallPaths::llvm_tool_path(LLVMTool tool) const
    -> std::filesystem::path {
  // TODO: Adjust this to work equally well on Windows.
  return prefix_ / "lib/carbon/llvm/bin" / std::string_view(tool.bin_name());
}

auto InstallPaths::clang_resource_path() const -> std::filesystem::path {
  // TODO: Adjust this to work equally well on Windows.
  return prefix_ / "lib/carbon/llvm/lib/clang/" CLANG_VERSION_MAJOR_STRING;
}

auto InstallPaths::llvm_runtime_srcs() const -> std::filesystem::path {
  // TODO: Adjust this to work equally well on Windows.
  return prefix_ / "lib/carbon/llvm/lib/clang/" CLANG_VERSION_MAJOR_STRING
                   "/src";
}

}  // namespace Carbon
