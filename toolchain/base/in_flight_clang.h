// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_IN_FLIGHT_CLANG_H
#define CARBON_TOOLCHAIN_BASE_IN_FLIGHT_CLANG_H

#include <thread>

#include "clang/AST/ASTContext.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"

namespace Carbon {
class InFlightClang {
 public:
  ~InFlightClang();

  // Runs the compiler on the passed code and stops it at a point suited for
  // doing any additional operations on the frontend.
  //
  // The arguments must produce exactly one compile job.
  static auto CompileFromArguments(
      llvm::ArrayRef<const char*> argv, llvm::StringRef target,
      llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
      std::unique_ptr<clang::DiagnosticConsumer> consumer)
      -> std::unique_ptr<InFlightClang>;

  auto getASTContext() -> clang::ASTContext&;
  auto getSourceManager() -> clang::SourceManager&;
  auto getSourceManager() const -> const clang::SourceManager&;
  auto getSema() -> clang::Sema&;

  // Take ownership of LLVMContext created by Clang.
  // Clang will still keep a reference to it and use it for code generation.
  auto takeLLVMContext() -> std::unique_ptr<llvm::LLVMContext>;

  // TODO: do not expose this and instead interact with Clang through Sema.
  auto getCodeGenerator() const -> clang::CodeGenerator&;

  auto finishCompilation() && -> std::unique_ptr<llvm::Module>;

 private:
  struct AstChannel;
  InFlightClang(clang::Sema* sema, clang::CodeGenAction* action,
                std::unique_ptr<AstChannel> chan, std::thread worker);

  clang::Sema* const sema_ = nullptr;
  clang::CodeGenAction* const action_ = nullptr;
  bool llvm_context_taken_ = false;

  std::unique_ptr<AstChannel> chan_;
  std::thread worker_;
};

}  // namespace Carbon

#endif
