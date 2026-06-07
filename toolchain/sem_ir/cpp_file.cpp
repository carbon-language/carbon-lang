// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/cpp_file.h"

#include "clang/AST/Mangle.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "common/check.h"

namespace Carbon::SemIR {

CppFile::CppFile(std::unique_ptr<clang::CompilerInstance> clang,
                 llvm::LLVMContext* llvm_context)
    : clang_(std::move(clang)), llvm_context_(llvm_context) {}

CppFile::~CppFile() = default;

auto CppFile::diagnostic_options() const -> const clang::DiagnosticOptions& {
  return clang_->getDiagnostics().getDiagnosticOptions();
}

auto CppFile::lang_options() const -> const clang::LangOptions& {
  return clang_->getLangOpts();
}

auto CppFile::source_manager() -> clang::SourceManager& {
  return clang_->getSourceManager();
}

auto CppFile::source_manager() const -> const clang::SourceManager& {
  return clang_->getSourceManager();
}

auto CppFile::diagnostics() const -> clang::DiagnosticsEngine& {
  return clang_->getDiagnostics();
}

auto CppFile::ast_context() -> clang::ASTContext& {
  return clang_->getASTContext();
}

auto CppFile::ast_context() const -> const clang::ASTContext& {
  return clang_->getASTContext();
}

auto CppFile::CreateMangleContext() -> void {
  CARBON_CHECK(!mangle_context_);
  mangle_context_.reset(ast_context().createMangleContext());
}

auto CppFile::mangle_context() const -> clang::MangleContext& {
  return *mangle_context_;
}

}  // namespace Carbon::SemIR
