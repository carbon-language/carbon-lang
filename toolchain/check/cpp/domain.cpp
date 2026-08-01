// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/domain.h"

#include "clang/Parse/Parser.h"

namespace Carbon::Check {

CppDomain::CppDomain(std::shared_ptr<clang::CompilerInstance> clang_instance,
                     std::unique_ptr<clang::Parser> parser,
                     llvm::ArrayRef<CppInputFile> inputs,
                     llvm::ArrayRef<clang::CodeGenerator*> code_generators,
                     llvm::LLVMContext* llvm_context)
    : clang_instance_(std::move(clang_instance)),
      parser_(std::move(parser)),
      llvm_context_(llvm_context) {
  CARBON_CHECK(inputs.size() == code_generators.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto res =
        code_generators_.Insert(inputs[i].check_ir_id, code_generators[i]);
    CARBON_CHECK(res.is_inserted(), "Duplicate CheckIRId {0} in CppDomain",
                 inputs[i].check_ir_id);
  }
}

CppDomain::~CppDomain() = default;

}  // namespace Carbon::Check
