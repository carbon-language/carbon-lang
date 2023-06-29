// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/codegen/codegen.h"

#include <cstdio>

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"

namespace Carbon {

auto CodeGen::CreateTargetMachine() -> llvm::TargetMachine* {
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
    error_stream_ << "ERROR: " << error << "\n";
    return nullptr;
  }

  constexpr llvm::StringLiteral CPU = "generic";
  constexpr llvm::StringLiteral Features = "";

  llvm::TargetOptions target_opts;
  std::optional<llvm::Reloc::Model> reloc_model;
  auto* target_machine = target->createTargetMachine(
      target_triple, CPU, Features, target_opts, reloc_model);
  return target_machine;
}

auto CodeGen::EmitCode(llvm::raw_pwrite_stream& dest,
                       llvm::TargetMachine* target_machine,
                       llvm::CodeGenFileType file_type) -> bool {
  Module.setDataLayout(target_machine->createDataLayout());
  Module.setTargetTriple(target_triple);

  // Using the legacy PM to generate the assembly since the new PM
  // does not work with this yet.
  // TODO: make the new PM work with the codegen pipeline.

  llvm::legacy::PassManager pass;

  if (target_machine->addPassesToEmitFile(pass, dest, nullptr, file_type)) {
    error_stream_ << "Nothing to write to object file\n";
    return false;
  }

  pass.run(Module);
  return true;
}

auto CodeGen::PrintAssembly() -> bool {
  auto* target_machine = CreateTargetMachine();
  if (target_machine == nullptr) {
    return false;
  }
  bool has_errors = EmitCode(output_stream_, target_machine,
                             llvm::CodeGenFileType::CGFT_AssemblyFile);
  delete target_machine;
  return has_errors;
}
auto CodeGen::GenerateObjectCode(llvm::StringRef output_file) -> bool {
  auto* target_machine = CreateTargetMachine();
  if (target_machine == nullptr) {
    return false;
  }
  std::error_code ec;
  llvm::raw_fd_ostream dest(output_file, ec, llvm::sys::fs::OF_None);
  if (ec) {
    error_stream_ << "Error: Could not open file: " << ec.message() << "\n";
    return false;
  }
  bool has_errors =
      EmitCode(dest, target_machine, llvm::CodeGenFileType::CGFT_ObjectFile);
  delete target_machine;
  return has_errors;
}
}  // namespace Carbon
