// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_DOMINANCE_H_
#define CARBON_TOOLCHAIN_SEM_IR_DOMINANCE_H_

#include "common/error.h"

namespace Carbon::SemIR {

class File;

// Verifies that every use of an instruction's value within a function body is
// dominated by an evaluation of that instruction, or that the instruction has a
// constant value. This is checked for every function definition in `file`, and
// for each resolved specific of a generic function, where spliced instructions
// can be resolved.
//
// Nothing is checked if `file` has errors, because these invariants don't
// necessarily hold for invalid IR.
auto VerifyDominance(const File& file) -> ErrorOr<Success>;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_DOMINANCE_H_
