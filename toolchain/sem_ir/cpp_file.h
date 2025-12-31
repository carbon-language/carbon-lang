// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_CPP_FILE_H_
#define CARBON_TOOLCHAIN_SEM_IR_CPP_FILE_H_

#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/FileSystem.h"

namespace Carbon::SemIR {

// The result of compiling the C++ portion of a `File`, including both any
// imported C++ headers and any inline C++ fragments.
class CppFile {
 public:
  explicit CppFile(std::unique_ptr<clang::CompilerInstance> clang)
      : clang_(std::move(clang)) {}

  // Access to compilation options.
  auto diagnostic_options() const -> const clang::DiagnosticOptions& {
    return clang_->getDiagnostics().getDiagnosticOptions();
  }
  auto lang_options() const -> const clang::LangOptions& {
    return clang_->getLangOpts();
  }

  // Access to Clang's compilation environment.
  auto source_manager() -> clang::SourceManager& {
    return clang_->getSourceManager();
  }
  auto source_manager() const -> const clang::SourceManager& {
    return clang_->getSourceManager();
  }
  // TODO: This doesn't really belong here, but is currently used by lowering
  // because Clang's code generation may produce diagnostics.
  auto diagnostics() const -> clang::DiagnosticsEngine& {
    return clang_->getDiagnostics();
  }

  // Access to layers of Clang's C++ representation.
  auto ast_context() -> clang::ASTContext& { return clang_->getASTContext(); }
  auto ast_context() const -> const clang::ASTContext& {
    return clang_->getASTContext();
  }

  // A list of all the top-level decl groups produced in this compilation.
  auto decl_groups() -> llvm::SmallVector<clang::DeclGroupRef>& {
    return decl_groups_;
  }
  auto decl_groups() const -> const llvm::SmallVector<clang::DeclGroupRef>& {
    return decl_groups_;
  }

 private:
  std::unique_ptr<clang::CompilerInstance> clang_;
  llvm::SmallVector<clang::DeclGroupRef> decl_groups_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_CPP_FILE_H_
