// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_SUGARED_TYPE_H_
#define CARBON_TOOLCHAIN_SEM_IR_SUGARED_TYPE_H_

#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// Returns an instruction describing the type of `inst_id`, preferring an
// instruction that reflects how the type was written in the source over the
// canonical instruction for the type.
//
// The type of an instruction is tracked as a `TypeId`, which is canonical, and
// so says nothing about how the type was spelled: for example, a type named by
// an alias is indistinguishable from the type that the alias names. However,
// the instructions that formed the type as written are usually still present,
// and can often be found by looking at the instruction that has the type,
// rather than at the type itself. For example, the type of a call to a
// non-generic function is the declared return type of that function, and the
// function tracks the instruction for its return type as written.
//
// This process is opportunistic: where we can't do better, the canonical type
// instruction is returned. The result always describes the same type as
// `sem_ir.insts().Get(inst_id).type_id()`; only the spelling can differ.
//
// Returns `None` if `inst_id` has no type.
auto GetSugaredTypeOfInst(const File& sem_ir, InstId inst_id) -> TypeInstId;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_SUGARED_TYPE_H_
