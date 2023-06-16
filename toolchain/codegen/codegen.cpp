// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/codegen/codegen.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"

namespace Carbon {
void CodeGen::generate_obj_file_from_module(llvm::Module& module) {
  // Initialize the target registry etc.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  auto target_triple = llvm::sys::getDefaultTargetTriple();

  std::string error;
  const auto* target = llvm::TargetRegistry::lookupTarget(target_triple, error);

  if (!target) {
    output << error;
    return;
  }

  const auto* cpu = "generic";
  const auto* features = "";

  llvm::TargetOptions target_opts;
  auto reloc_model = std::optional<llvm::Reloc::Model>();
  auto* target_machine = target->createTargetMachine(
      target_triple, cpu, features, target_opts, reloc_model);
  module.setDataLayout(target_machine->createDataLayout());
  module.setTargetTriple(target_triple);

  const auto* filename = "objfile.o";
  std::error_code ec;
  llvm::raw_fd_ostream dest(filename, ec, llvm::sys::fs::OF_None);

  if (ec) {
    output << "Could not open file: " << ec.message();
    return;
  }

  llvm::legacy::PassManager pass;
  auto file_type = llvm::CGFT_ObjectFile;

  if (target_machine->addPassesToEmitFile(pass, dest, nullptr, file_type)) {
    output << "Could not write to object file\n";
    return;
  }

  pass.run(module);
  dest.flush();
  delete target_machine;
}
}  // namespace Carbon
