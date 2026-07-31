// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_CPP_FILE_H_
#define CARBON_TOOLCHAIN_SEM_IR_CPP_FILE_H_

#include <memory>

namespace clang {
class ASTContext;
class CodeGenerator;
class CompilerInstance;
class DiagnosticOptions;
class DiagnosticsEngine;
class LangOptions;
class MangleContext;
class SourceManager;
}  // namespace clang

namespace llvm {
class LLVMContext;
}  // namespace llvm

namespace Carbon::SemIR {

// The result of compiling the C++ portion of a `File`, including both any
// imported C++ headers and any inline C++ fragments.
class CppFile {
 public:
  explicit CppFile(std::shared_ptr<clang::CompilerInstance> clang,
                   std::unique_ptr<clang::MangleContext> mangle_context,
                   llvm::LLVMContext* llvm_context,
                   clang::CodeGenerator* code_generator);
  ~CppFile();

  // Access to compilation options.
  auto diagnostic_options() const -> const clang::DiagnosticOptions&;
  auto lang_options() const -> const clang::LangOptions&;

  // Access to Clang's compilation environment.
  auto source_manager() -> clang::SourceManager&;
  auto source_manager() const -> const clang::SourceManager&;
  // TODO: This doesn't really belong here, but is currently used by lowering
  // because Clang's code generation may produce diagnostics.
  auto diagnostics() const -> clang::DiagnosticsEngine&;

  // Access to layers of Clang's C++ representation.
  auto ast_context() -> clang::ASTContext&;
  auto ast_context() const -> const clang::ASTContext&;

  auto mangle_context() const -> clang::MangleContext&;

  auto llvm_context() const -> llvm::LLVMContext* { return llvm_context_; }
  auto code_generator() const -> clang::CodeGenerator* {
    return code_generator_;
  }

 private:
  std::shared_ptr<clang::CompilerInstance> clang_;
  std::unique_ptr<clang::MangleContext> mangle_context_;
  llvm::LLVMContext* llvm_context_;
  clang::CodeGenerator* code_generator_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_CPP_FILE_H_
