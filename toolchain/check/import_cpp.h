// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_IMPORT_CPP_H_
#define CARBON_TOOLCHAIN_CHECK_IMPORT_CPP_H_

#include "llvm/ADT/StringRef.h"
#include "toolchain/check/context.h"
#include "toolchain/check/diagnostic_helpers.h"

namespace Carbon::Check {

// Generates a C++ header that includes the imported cpp files, parses it and
// report errors and warnings. If successful, adds a `Cpp` namespace.
auto ImportCppFiles(Context& context, llvm::StringRef importing_file_path,
                    llvm::ArrayRef<Parse::Tree::PackagingNames> imports,
                    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_IMPORT_CPP_H_
