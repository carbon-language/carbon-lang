// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/domain.h"

#include "clang/Parse/Parser.h"

namespace Carbon::Check {

CppDomain::CppDomain(std::shared_ptr<clang::CompilerInstance> clang_instance,
                     std::unique_ptr<clang::Parser> parser,
                     llvm::ArrayRef<clang::CodeGenerator*> code_generators,
                     llvm::LLVMContext* llvm_context)
    : clang_instance_(std::move(clang_instance)),
      parser_(std::move(parser)),
      code_generators_(code_generators.begin(), code_generators.end()),
      llvm_context_(llvm_context) {}

CppDomain::~CppDomain() = default;

}  // namespace Carbon::Check
