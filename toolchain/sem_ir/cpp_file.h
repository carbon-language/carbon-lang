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
  explicit CppFile(std::unique_ptr<clang::CompilerInstance> clang,
                   llvm::LLVMContext* llvm_context);
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

  // Creates the mangle context for this file's C++ AST. Must be called once the
  // AST context is available (after the frontend begins the source file) and
  // before `mangle_context()` is used.
  auto CreateMangleContext() -> void;
  auto mangle_context() const -> clang::MangleContext&;

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
  // Created by `CreateMangleContext()` once the AST context is available.
  std::unique_ptr<clang::MangleContext> mangle_context_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_CPP_FILE_H_
