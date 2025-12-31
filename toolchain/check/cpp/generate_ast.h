// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_GENERATE_AST_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_GENERATE_AST_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "toolchain/check/context.h"

namespace Carbon::Check {

// Generates a Clang AST for the given C++ imports and sets it as the context's
// `cpp_context` and the SemIR's `cpp_file`. Returns a bool that represents
// whether compilation was successful.
auto GenerateAst(Context& context,
                 llvm::ArrayRef<Parse::Tree::PackagingNames> imports,
                 llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
                 std::shared_ptr<clang::CompilerInvocation> base_invocation)
    -> bool;

// Finishes AST generation for the given checking context. Performs end of file
// steps such as template instantiation and warning on unused declarations.
auto FinishAst(Context& context) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_GENERATE_AST_H_
