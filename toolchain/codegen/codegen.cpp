// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/codegen/codegen.h"

#include <cstdio>
#include <memory>

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"

namespace Carbon {

auto CodeGen::Create(llvm::Module& module, llvm::StringRef target_triple,
                     llvm::raw_ostream& errors) -> std::optional<CodeGen> {
  // Initialize the target registry etc.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  std::string error;
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget(target_triple, error);

  if (!target) {
    errors << "ERROR: Invalid target: " << error << "\n";
    return {};
  }
  module.setTargetTriple(target_triple);

  constexpr llvm::StringLiteral CPU = "generic";
  constexpr llvm::StringLiteral Features = "";

  llvm::TargetOptions target_opts;
  std::optional<llvm::Reloc::Model> reloc_model;
  CodeGen codegen(module, errors);
  codegen.target_machine_.reset(target->createTargetMachine(
      target_triple, CPU, Features, target_opts, reloc_model));
  return codegen;
}

auto CodeGen::EmitAssembly(llvm::raw_pwrite_stream& out) -> bool {
  return EmitCode(out, llvm::CodeGenFileType::CGFT_AssemblyFile);
}

auto CodeGen::EmitObject(llvm::raw_pwrite_stream& out) -> bool {
  return EmitCode(out, llvm::CodeGenFileType::CGFT_ObjectFile);
}

auto CodeGen::EmitCode(llvm::raw_pwrite_stream& out,
                       llvm::CodeGenFileType file_type) -> bool {
  module_.setDataLayout(target_machine_->createDataLayout());

  // Using the legacy PM to generate the assembly since the new PM
  // does not work with this yet.
  // TODO: make the new PM work with the codegen pipeline.
  llvm::legacy::PassManager pass;
  // Note that this returns true on an error.
  if (target_machine_->addPassesToEmitFile(pass, out, nullptr, file_type)) {
    errors_ << "ERROR: Unable to emit to this file.\n";
    return false;
  }

  pass.run(module_);
  return true;
}

}  // namespace Carbon
