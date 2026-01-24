// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/codegen/codegen.h"

#include <memory>
#include <optional>
#include <string>

#include "common/check.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "toolchain/diagnostics/diagnostic_consumer.h"

namespace Carbon {

auto CodeGen::EmitAssembly(llvm::raw_pwrite_stream& out) -> bool {
  return EmitCode(out, llvm::CodeGenFileType::AssemblyFile);
}

auto CodeGen::EmitObject(llvm::raw_pwrite_stream& out) -> bool {
  return EmitCode(out, llvm::CodeGenFileType::ObjectFile);
}

auto CodeGen::EmitCode(llvm::raw_pwrite_stream& out,
                       llvm::CodeGenFileType file_type) -> bool {
  module_->setDataLayout(target_machine_->createDataLayout());

  // Using the legacy PM to generate the assembly since the new PM
  // does not work with this yet.
  // TODO: Make the new PM work with the codegen pipeline.
  llvm::legacy::PassManager pass;
  // Note that this returns true on an error.
  if (target_machine_->addPassesToEmitFile(pass, out, nullptr, file_type)) {
    CARBON_DIAGNOSTIC(CodeGenUnableToEmit, Error,
                      "unable to emit to this file");
    emitter_.Emit(module_->getName(), CodeGenUnableToEmit);
    return false;
  }

  pass.run(*module_);
  return true;
}

}  // namespace Carbon
