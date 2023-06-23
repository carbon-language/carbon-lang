// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CODEGEN_CODEGEN_H_
#define CARBON_TOOLCHAIN_CODEGEN_CODEGEN_H_

#include <cstdint>

#include "llvm/IR/Module.h"

namespace Carbon {
// Prints the assembly to stdout for the given llvm module.
auto PrintAssemblyFromModule(llvm::Module& module,
                             llvm::StringRef target_triple,
                             llvm::raw_pwrite_stream& error_stream,
                             llvm::raw_pwrite_stream& output_stream) -> bool;
}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_CODEGEN_CODEGEN_H_
