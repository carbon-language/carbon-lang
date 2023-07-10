// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CODEGEN_CODEGEN_H_
#define CARBON_TOOLCHAIN_CODEGEN_CODEGEN_H_

#include <cstdint>

#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"

namespace Carbon {

class CodeGen {
 public:
  static auto Create(llvm::Module& module, llvm::StringRef target_triple,
                     llvm::raw_ostream& errors) -> std::optional<CodeGen>;

  // Generates the object code file.
  // Returns false in case of failure, and any information about the failure is
  // printed to the error stream.
  auto EmitObject(llvm::raw_pwrite_stream& out) -> bool;

  // Prints the assembly to stdout.
  // Returns false in case of failure, and any information about the failure is
  // printed to the error stream.
  auto EmitAssembly(llvm::raw_pwrite_stream& out) -> bool;

 private:
  explicit CodeGen(llvm::Module& module, llvm::raw_ostream& errors)
      : module_(module), errors_(errors) {}

  // Using the llvm pass emits either assembly or object code to dest.
  // Returns false in case of failure, and any information about the failure is
  // printed to the error stream.
  auto EmitCode(llvm::raw_pwrite_stream& out, llvm::CodeGenFileType file_type)
      -> bool;

  llvm::Module& module_;
  llvm::raw_ostream& errors_;
  std::unique_ptr<llvm::TargetMachine> target_machine_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_CODEGEN_CODEGEN_H_
