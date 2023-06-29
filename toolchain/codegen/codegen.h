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
  CodeGen(llvm::Module& module, llvm::StringRef triple,
          llvm::raw_pwrite_stream& error_stream,
          llvm::raw_pwrite_stream& output_stream)
      : Module(module),
        output_stream_(output_stream),
        error_stream_(error_stream),
        target_triple(triple){};

  // Generates the object code file.
  // Returns false in case of failure, and any information about the failure is
  // printed to the error stream.
  auto GenerateObjectCode(llvm::StringRef output_file) -> bool;

  // Prints the assembly to stdout.
  // Returns false in case of failure, and any information about the failure is
  // printed to the error stream.
  auto PrintAssembly() -> bool;

 private:
  llvm::Module& Module;
  llvm::raw_pwrite_stream& output_stream_;
  llvm::raw_pwrite_stream& error_stream_;
  llvm::StringRef target_triple;

  // Creates the target machine for triple.
  // Returns nullptr in case of failure, and any information about the failure
  // is printed to the error stream.
  auto CreateTargetMachine() -> std::unique_ptr<llvm::TargetMachine>;

  // Using the llvm pass emits either assembly or object code to dest.
  // Returns false in case of failure, and any information about the failure is
  // printed to the error stream.
  auto EmitCode(llvm::raw_pwrite_stream& dest,
                llvm::TargetMachine* target_machine,
                llvm::CodeGenFileType file_type) -> bool;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_CODEGEN_CODEGEN_H_
