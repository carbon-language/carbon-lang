// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_GENERATE_AST_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_GENERATE_AST_H_

#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "toolchain/check/context.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Generates a Clang AST for the given C++ imports and sets it as the context's
// `cpp_context` and the SemIR's `cpp_file`. Returns a bool that represents
// whether compilation was successful.
auto GenerateAst(Context& context,
                 llvm::ArrayRef<Parse::Tree::PackagingNames> imports,
                 llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
                 llvm::LLVMContext* llvm_context,
                 std::shared_ptr<clang::CompilerInvocation> base_invocation)
    -> bool;

// Injects C++ code from `inline Cpp` into the active Clang AST context.
// Returns a bool representing whether parsing was successful.
auto InjectAstFromInlineCode(Context& context, SemIR::LocId loc_id,
                             llvm::StringRef source_code) -> void;

// Finishes AST generation for the given checking context. Performs end of file
// steps such as template instantiation and warning on unused declarations.
auto FinishAst(Context& context) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_GENERATE_AST_H_
