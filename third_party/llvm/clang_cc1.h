// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_THIRD_PARTY_LLVM_CLANG_CC1_H_
#define CARBON_THIRD_PARTY_LLVM_CLANG_CC1_H_

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "toolchain/base/install_paths.h"

namespace Carbon {

// Emulates Clang's `cc1_main` but in a way that doesn't assume it is running
// in the main thread and can more easily fit into library calls to do
// compiles.
//
// TODO: Much of the logic here should be factored out of the CC1
// implementation in Clang's driver and into a reusable part of its libraries.
// That should allow reducing the code here to a minimal amount.
auto RunClangCC1Main(const InstallPaths& installation,
                     llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
                     llvm::SmallVectorImpl<const char*>& cc1_args,
                     bool enable_leaking) -> int;

}  // namespace Carbon

#endif  // CARBON_THIRD_PARTY_LLVM_CLANG_CC1_H_
