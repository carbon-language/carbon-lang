// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/codegen/codegen.h"

#include <cstdio>

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"

namespace Carbon {

void PrintAssemblyFromModule(llvm::Module& module,
                             llvm::StringRef target_triple,
                             llvm::raw_ostream& error_stream) {
  // Initialize the target registry etc.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  std::string error;
  llvm::StringRef triple = target_triple;
  std::string host_triple;
  if (target_triple.empty()) {
    host_triple = llvm::sys::getDefaultTargetTriple();
    triple = host_triple;
  }
  const auto* target = llvm::TargetRegistry::lookupTarget(triple, error);

  if (!target) {
    error_stream << error;
    return;
  }

  constexpr llvm::StringLiteral CPU = "generic";
  constexpr llvm::StringLiteral Features = "";

  llvm::TargetOptions target_opts;
  std::optional<llvm::Reloc::Model> reloc_model;
  auto* target_machine = target->createTargetMachine(
      target_triple, CPU, Features, target_opts, reloc_model);
  module.setDataLayout(target_machine->createDataLayout());
  module.setTargetTriple(target_triple);

  // Using the legacy PM to generate the assembly since the new PM
  // does not work with this yet.
  // FIXME: make the new PM work with the codegen pipeline.

  llvm::legacy::PassManager pass;
  auto file_type = llvm::CGFT_AssemblyFile;

  if (target_machine->addPassesToEmitFile(pass, llvm::outs(), nullptr,
                                          file_type)) {
    error_stream << "Nothing to write to object file\n";
    return;
  }

  pass.run(module);
  delete target_machine;
}
}  // namespace Carbon
