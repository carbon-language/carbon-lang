// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_CPP_DOMAIN_H_
#define CARBON_TOOLCHAIN_SEM_IR_CPP_DOMAIN_H_

#include <memory>

#include "common/check.h"
#include "common/map.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "toolchain/sem_ir/ids.h"

namespace clang {
class CodeGenerator;
class CompilerInstance;
class Module;
class Parser;
}  // namespace clang

namespace llvm {
class LLVMContext;
}  // namespace llvm

namespace Carbon::SemIR {

// An input Carbon file and its CheckIRId for C++ domain code generation.
struct CppInputFile {
  // The ID used to identify this file within SemIR.
  CheckIRId check_ir_id;
  // The Carbon source filename for this input.
  llvm::StringRef filename;
  // Whether this input IR will be lowered. If not, we don't need to build a
  // Clang CodeGenerator for it.
  bool is_lowered;
};

// A C++ compilation domain, including a live Clang instance that can be used to
// parse more code into that domain. May be shared across multiple Carbon files.
class CppDomain {
 public:
  explicit CppDomain(std::shared_ptr<clang::CompilerInstance> clang_instance,
                     std::unique_ptr<clang::Parser> parser,
                     llvm::ArrayRef<CppInputFile> inputs,
                     llvm::ArrayRef<clang::CodeGenerator*> code_generators,
                     llvm::LLVMContext* llvm_context);
  ~CppDomain();

  auto clang_instance() const -> clang::CompilerInstance& {
    return *clang_instance_;
  }
  auto clang_instance_ptr() const -> std::shared_ptr<clang::CompilerInstance> {
    return clang_instance_;
  }
  auto parser() const -> clang::Parser& { return *parser_; }
  auto llvm_context() const -> llvm::LLVMContext* { return llvm_context_; }

  auto GetCodeGenerator(CheckIRId check_ir_id) const -> clang::CodeGenerator* {
    auto res = code_generators_.Lookup(check_ir_id);
    CARBON_CHECK(res, "No CodeGenerator found for CheckIRId {0}", check_ir_id);
    return res.value();
  }

  // Gets the Clang module corresponding to the given Carbon file, if one was
  // created.
  auto GetModule(CheckIRId check_ir_id) const -> clang::Module* {
    auto res = modules_.Lookup(check_ir_id);
    return res ? res.value() : nullptr;
  }

  // Associates a Clang module with a Carbon file.
  auto SetModule(CheckIRId check_ir_id, clang::Module* module) -> void {
    auto res = modules_.Insert(check_ir_id, module);
    CARBON_CHECK(res.is_inserted(), "Duplicate CheckIRId {0} in CppDomain",
                 check_ir_id);
  }

  auto header_modules() -> llvm::StringMap<clang::Module*>& {
    return header_modules_;
  }

 private:
  std::shared_ptr<clang::CompilerInstance> clang_instance_;
  std::unique_ptr<clang::Parser> parser_;
  Map<CheckIRId, clang::CodeGenerator*> code_generators_;
  Map<CheckIRId, clang::Module*> modules_;
  llvm::StringMap<clang::Module*> header_modules_;
  llvm::LLVMContext* llvm_context_ = nullptr;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_CPP_DOMAIN_H_
