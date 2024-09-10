// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_DECL_H_
#define CARBON_TOOLCHAIN_SEM_IR_DECL_H_

#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// Information about a declaration.
struct DeclInfo {
  // The pattern block, containing the pattern insts for the parameter pattern.
  InstBlockId pattern_block_id;
  // The declaration block, containing the declaration's parameters and their
  // types.
  InstBlockId decl_block_id;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_DECL_H_
