// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_CONTEXT_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_CONTEXT_H_

#include <memory>

#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/ASTUnit.h"
#include "llvm/ADT/SmallVector.h"

namespace Carbon::Check {

// Context for C++ code during check.
//
// This stores state for a Clang AST and Sema, as well as any additional
// information needed to perform mapping between Carbon and C++ types,
// declarations, and similar values.
class CppContext {
 public:
  explicit CppContext(clang::ASTUnit* ast_unit);
  ~CppContext();

  auto ast_context() -> clang::ASTContext& {
    return ast_unit_->getASTContext();
  }
  auto sema() -> clang::Sema& { return ast_unit_->getSema(); }

  auto clang_mangle_context() -> clang::MangleContext&;

  auto carbon_file_locations() -> llvm::SmallVector<clang::SourceLocation>& {
    return carbon_file_locations_;
  }

 private:
  // The ASTUnit is owned by the `CppFile`.
  clang::ASTUnit* ast_unit_;

  // Per-Carbon-file start locations for corresponding Clang source buffers.
  // Owned and managed by code in location.cpp.
  llvm::SmallVector<clang::SourceLocation> carbon_file_locations_;

  // The Clang mangle context for the target in the ASTContext.
  std::unique_ptr<clang::MangleContext> clang_mangle_context_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_CONTEXT_H_
