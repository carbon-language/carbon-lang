// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_CONTEXT_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_CONTEXT_H_

#include <memory>

#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Parse/Parser.h"
#include "common/check.h"
#include "llvm/ADT/SmallVector.h"

namespace Carbon::Check {

// Context for C++ code during check.
//
// This stores state for a Clang AST and Sema, as well as any additional
// information needed to perform mapping between Carbon and C++ types,
// declarations, and similar values.
class CppContext {
 public:
  explicit CppContext(clang::CompilerInstance& instance,
                      std::unique_ptr<clang::Parser> parser);
  ~CppContext();

  auto ast_context() -> clang::ASTContext& { return *ast_context_; }
  auto sema() -> clang::Sema& { return *sema_; }
  auto parser() -> clang::Parser& { return *parser_; }

  auto clang_mangle_context() -> clang::MangleContext&;

  auto carbon_file_locations() -> llvm::SmallVector<clang::SourceLocation>& {
    return carbon_file_locations_;
  }

 private:
  // The Clang AST context.
  clang::ASTContext* ast_context_;

  // The Clang semantic analysis engine.
  clang::Sema* sema_;

  // The Clang parser.
  std::unique_ptr<clang::Parser> parser_;

  // Per-Carbon-file start locations for corresponding Clang source buffers.
  // Owned and managed by code in location.cpp.
  llvm::SmallVector<clang::SourceLocation> carbon_file_locations_;

  // The Clang mangle context for the target in the ASTContext.
  std::unique_ptr<clang::MangleContext> clang_mangle_context_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_CONTEXT_H_
