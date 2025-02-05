// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_ABSOLUTE_NODE_ID_H_
#define CARBON_TOOLCHAIN_SEM_IR_ABSOLUTE_NODE_ID_H_

#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// A specific node location in a file.
struct AbsoluteNodeId {
  CheckIRId check_ir_id;
  Parse::NodeId node_id;
};

// Resolves the `InstId` to a series of `NodeId`s, which may be in different
// files. The vector will have one entry if there were no imports, and multiple
// entries when imports are traversed. The final entry is the actual
// declaration.
auto GetAbsoluteNodeId(const File* sem_ir, InstId inst_id)
    -> llvm::SmallVector<AbsoluteNodeId>;

// Similar to to above overload, but starting at a `LocId`.
auto GetAbsoluteNodeId(const File* sem_ir, LocId loc_id)
    -> llvm::SmallVector<AbsoluteNodeId>;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_ABSOLUTE_NODE_ID_H_
