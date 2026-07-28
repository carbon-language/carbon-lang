// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_DOMAIN_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_DOMAIN_H_

#include <memory>

namespace clang {
class CodeGenerator;
class CompilerInstance;
class Parser;
}  // namespace clang

namespace llvm {
class LLVMContext;
}  // namespace llvm

namespace Carbon::Check {

// A C++ compilation domain, including a live Clang instance that can be used to
// parse more code into that domain. May be shared across multiple Carbon files.
class CppDomain {
 public:
  explicit CppDomain(std::shared_ptr<clang::CompilerInstance> clang_instance,
                     std::unique_ptr<clang::Parser> parser,
                     clang::CodeGenerator* code_generator,
                     llvm::LLVMContext* llvm_context);
  ~CppDomain();

  auto clang_instance() const -> clang::CompilerInstance& {
    return *clang_instance_;
  }
  auto clang_instance_ptr() const -> std::shared_ptr<clang::CompilerInstance> {
    return clang_instance_;
  }
  auto parser() const -> clang::Parser& { return *parser_; }
  auto code_generator() const -> clang::CodeGenerator* {
    return code_generator_;
  }
  auto llvm_context() const -> llvm::LLVMContext* { return llvm_context_; }

 private:
  std::shared_ptr<clang::CompilerInstance> clang_instance_;
  std::unique_ptr<clang::Parser> parser_;
  clang::CodeGenerator* code_generator_ = nullptr;
  llvm::LLVMContext* llvm_context_ = nullptr;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_DOMAIN_H_
