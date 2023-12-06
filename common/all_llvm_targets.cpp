// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/init_llvm.h"
#include "llvm/Support/TargetSelect.h"

namespace Carbon {

static auto InitLLVMTargets() -> void {
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();
}

char InitLLVM::RegisterTargets = (InitializeTargets = &InitLLVMTargets, 0);

}  // namespace Carbon
