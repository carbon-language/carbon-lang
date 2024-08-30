// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/install/install_paths_test_helpers.h"

#include "testing/base/global_exe_path.h"

namespace Carbon::Testing {

// Prepares the VFS with prelude files from the real filesystem. Primarily for
// tests.
auto CopyPreludeFiles(InstallPaths install_paths,
                      llvm::vfs::InMemoryFileSystem* vfs) -> void {
  // Load the prelude into the test VFS.
  auto real_fs = llvm::vfs::getRealFileSystem();
  auto prelude = install_paths.FindPreludeFiles();
  CARBON_CHECK(prelude.ok()) << prelude.error();

  for (const auto& path : *prelude) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file =
        real_fs->getBufferForFile(path);
    CARBON_CHECK(file) << "Error getting file: " << file.getError().message();
    bool added = vfs->addFile(path, /*ModificationTime=*/0, std::move(*file));
    CARBON_CHECK(added) << "Duplicate file: " << path;
  }
}

}  // namespace Carbon::Testing
