// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_IMPORT_CPP_H_
#define CARBON_TOOLCHAIN_CHECK_IMPORT_CPP_H_

#include "llvm/ADT/StringRef.h"
#include "toolchain/check/context.h"
#include "toolchain/check/diagnostic_helpers.h"

namespace Carbon::Check {

// Parses the C++ code and report errors and warnings.
auto ImportCppFile(Context& context, SemIRLoc loc,
                   llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
                   llvm::StringRef file_path, llvm::StringRef code) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_IMPORT_CPP_H_
