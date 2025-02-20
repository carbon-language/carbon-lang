// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/init_llvm.h"
#include "llvm/Support/TargetSelect.h"

namespace Carbon {

static auto InitLlvmTargets() -> void {
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();
}

// On program startup, set `InitLlvm::InitializeTargets` to be our
// initialization function so that `InitLlvm` can call it at the right moment.
const char InitLlvm::RegisterTargets =
    (InitializeTargets = &InitLlvmTargets, 0);

}  // namespace Carbon
