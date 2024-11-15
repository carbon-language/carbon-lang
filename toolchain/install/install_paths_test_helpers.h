// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_INSTALL_INSTALL_PATHS_TEST_HELPERS_H_
#define CARBON_TOOLCHAIN_INSTALL_INSTALL_PATHS_TEST_HELPERS_H_

#include "llvm/Support/VirtualFileSystem.h"
#include "toolchain/install/install_paths.h"

namespace Carbon::Testing {

// Prepares the VFS with prelude files from the real filesystem.
auto AddPreludeFilesToVfs(
    InstallPaths install_paths,
    llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem>& vfs) -> void;

}  // namespace Carbon::Testing

#endif  // CARBON_TOOLCHAIN_INSTALL_INSTALL_PATHS_TEST_HELPERS_H_
