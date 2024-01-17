// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_INST_PROFILE_H_
#define CARBON_TOOLCHAIN_SEM_IR_INST_PROFILE_H_

#include "llvm/ADT/FoldingSet.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::SemIR {

// Computes a profile of the given constant instruction, for canonicalization /
// deduplication.
auto ProfileConstant(llvm::FoldingSetNodeID& id, const File& sem_ir, Inst inst)
    -> void;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_INST_PROFILE_H_
