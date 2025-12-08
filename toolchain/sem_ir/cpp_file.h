// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_CPP_FILE_H_
#define CARBON_TOOLCHAIN_SEM_IR_CPP_FILE_H_

#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/FileSystem.h"

namespace Carbon::SemIR {

// The result of compiling the C++ portion of a `File`, including both any
// imported C++ headers and any inline C++ fragments.
class CppFile {
 public:
  explicit CppFile(std::unique_ptr<clang::ASTUnit> ast_unit)
      : ast_unit_(std::move(ast_unit)) {}

  // Access to compilation options.
  auto diagnostic_options() const -> const clang::DiagnosticOptions& {
    return ast_unit_->getDiagnostics().getDiagnosticOptions();
  }
  auto lang_options() const -> const clang::LangOptions& {
    return ast_unit_->getLangOpts();
  }

  // Access to Clang's compilation environment.
  auto source_manager() -> clang::SourceManager& {
    return ast_unit_->getSourceManager();
  }
  auto source_manager() const -> const clang::SourceManager& {
    return ast_unit_->getSourceManager();
  }
  // TODO: This doesn't really belong here, but is currently used by lowering
  // because Clang's code generation may produce diagnostics.
  auto diagnostics() const -> clang::DiagnosticsEngine& {
    return ast_unit_->getDiagnostics();
  }

  // Access to layers of Clang's C++ representation.
  auto ast_context() -> clang::ASTContext& {
    return ast_unit_->getASTContext();
  }
  auto ast_context() const -> const clang::ASTContext& {
    return ast_unit_->getASTContext();
  }

  // Visit all top-level declarations in the file.
  auto VisitLocalTopLevelDecls(
      llvm::function_ref<auto(const clang::Decl*)->void> visitor) const -> void;

 private:
  std::unique_ptr<clang::ASTUnit> ast_unit_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_CPP_FILE_H_
