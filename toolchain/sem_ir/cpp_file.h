// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_CPP_FILE_H_
#define CARBON_TOOLCHAIN_SEM_IR_CPP_FILE_H_

#include "clang/Basic/CodeGenOptions.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"

namespace Carbon::SemIR {

// The result of compiling the C++ portion of a `File`, including both any
// imported C++ headers and any inline C++ fragments.
class CppFile {
 public:
  explicit CppFile(std::unique_ptr<clang::CompilerInstance> clang,
                   llvm::LLVMContext* llvm_context)
      : clang_(std::move(clang)), llvm_context_(llvm_context) {}

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

  auto llvm_context() const -> llvm::LLVMContext* { return llvm_context_; }
  auto SetCodeGenerator(clang::CodeGenerator* code_generator) -> void {
    code_generator_ = code_generator;
  }
  auto GetCodeGenerator() const -> clang::CodeGenerator* {
    // Clang code generation should not actually modify the AST, but isn't
    // const-correct.
    return code_generator_;
  }

 private:
  std::unique_ptr<clang::CompilerInstance> clang_;
  llvm::LLVMContext* llvm_context_;
  clang::CodeGenerator* code_generator_ = nullptr;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_CPP_FILE_H_
