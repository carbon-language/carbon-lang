// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/init_llvm_with_targets.h"

#include "llvm/Support/TargetSelect.h"

namespace Carbon {

InitLLVMWithTargets::InitLLVMWithTargets(int& argc, char**& argv)
    : init_llvm(argc, argv) {
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();
}

}  // namespace Carbon
