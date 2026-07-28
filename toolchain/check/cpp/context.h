// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_CONTEXT_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_CONTEXT_H_

#include <memory>

#include "clang/Basic/SourceLocation.h"
#include "common/check.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/check/cpp/diagnostic_listener.h"
#include "toolchain/check/cpp/domain.h"

namespace clang {
class ASTContext;
class CompilerInstance;
class FunctionDecl;
class MangleContext;
class Parser;
class Sema;
}  // namespace clang

namespace Carbon::Check {

// Context for C++ code during check.
//
// This stores state for a Clang AST and Sema, as well as any additional
// information needed to perform mapping between Carbon and C++ types,
// declarations, and similar values.
class CppContext {
 public:
  explicit CppContext(std::shared_ptr<CppDomain> domain,
                      std::unique_ptr<CppDiagnosticListener> listener);
  ~CppContext();

  auto ast_context() -> clang::ASTContext&;
  auto sema() -> clang::Sema&;
  auto parser() -> clang::Parser& { return domain_->parser(); }

  auto domain() -> CppDomain& { return *domain_; }
  auto domain() const -> const CppDomain& { return *domain_; }
  auto domain_ptr() const -> std::shared_ptr<CppDomain> { return domain_; }

  auto clang_mangle_context() -> clang::MangleContext&;

  auto carbon_file_locations() -> llvm::SmallVector<clang::SourceLocation>& {
    return carbon_file_locations_;
  }

  auto placement_new_decl() const -> clang::FunctionDecl* {
    return placement_new_decl_;
  }
  void set_placement_new_decl(clang::FunctionDecl* decl) {
    placement_new_decl_ = decl;
  }

 private:
  // The C++ compilation domain.
  std::shared_ptr<CppDomain> domain_;

  // TODO: All of the below state that is not specific to a particular
  // Check::Context or SemIR::File should be moved into CppDomain.

  // Per-Carbon-file start locations for corresponding Clang source buffers.
  // Owned and managed by code in location.cpp.
  llvm::SmallVector<clang::SourceLocation> carbon_file_locations_;

  // The Clang mangle context for the target in the ASTContext.
  std::unique_ptr<clang::MangleContext> clang_mangle_context_;

  // The cached placement new function declaration.
  clang::FunctionDecl* placement_new_decl_ = nullptr;

  // Listener for Clang diagnostics while checking this Carbon context.
  std::unique_ptr<CppDiagnosticListener> diagnostic_listener_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_CONTEXT_H_
