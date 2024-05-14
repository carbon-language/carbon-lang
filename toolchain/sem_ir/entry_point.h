// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_ENTRY_POINT_H_
#define CARBON_TOOLCHAIN_SEM_IR_ENTRY_POINT_H_

#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

// Returns whether the specified function is the entry point function for the
// program, `Main.Run`.
auto IsEntryPoint(const SemIR::File& file, SemIR::FunctionId function_id)
    -> bool;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_ENTRY_POINT_H_
