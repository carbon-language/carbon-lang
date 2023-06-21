// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CODEGEN_CODEGEN_H_
#define CARBON_TOOLCHAIN_CODEGEN_CODEGEN_H_

#include <cstdint>

#include "llvm/IR/Module.h"

namespace Carbon {
// llvm::raw_ostream& output = llvm::outs();
void PrintDisassemblyFromModule(llvm::Module& module);
}  // namespace Carbon
#endif  // CARBON_TOOLCHAIN_CODEGEN_CODEGEN_H_
