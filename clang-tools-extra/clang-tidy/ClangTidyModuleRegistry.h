//===--- ClangTidyModuleRegistry.h - clang-tidy -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANG_TIDY_MODULE_REGISTRY_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANG_TIDY_MODULE_REGISTRY_H

#include "ClangTidyModule.h"
#include "llvm/Support/Registry.h"

extern template class llvm::Registry<clang::tidy::ClangTidyModule>;

namespace clang {
namespace tidy {

typedef llvm::Registry<ClangTidyModule> ClangTidyModuleRegistry;

} // end namespace tidy
} // end namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANG_TIDY_MODULE_REGISTRY_H
