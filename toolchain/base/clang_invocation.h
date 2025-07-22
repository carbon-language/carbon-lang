// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_CLANG_INVOCATION_H_
#define CARBON_TOOLCHAIN_BASE_CLANG_INVOCATION_H_

#include <string>

#include "clang/Frontend/CompilerInvocation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon {

// Builds and returns a clang `CompilerInvocation` to use when building code for
// interop, from a list of clang driver arguments. Emits diagnostics to
// `consumer` if the arguments are invalid.
auto BuildClangInvocation(Diagnostics::Consumer& consumer,
                          llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
                          llvm::ArrayRef<std::string> clang_path_and_args)
    -> std::unique_ptr<clang::CompilerInvocation>;

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_CLANG_INVOCATION_H_
