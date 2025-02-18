// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CONTROL_FLOW_H_
#define CARBON_TOOLCHAIN_CHECK_CONTROL_FLOW_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Adds a `Branch` instruction branching to a new instruction block, and
// returns the ID of the new block. All paths to the branch target must go
// through the current block, though not necessarily through this branch.
auto AddDominatedBlockAndBranch(Context& context, Parse::NodeId node_id)
    -> SemIR::InstBlockId;

// Adds a `Branch` instruction branching to a new instruction block with a
// value, and returns the ID of the new block. All paths to the branch target
// must go through the current block.
auto AddDominatedBlockAndBranchWithArg(Context& context, Parse::NodeId node_id,
                                       SemIR::InstId arg_id)
    -> SemIR::InstBlockId;

// Adds a `BranchIf` instruction branching to a new instruction block, and
// returns the ID of the new block. All paths to the branch target must go
// through the current block.
auto AddDominatedBlockAndBranchIf(Context& context, Parse::NodeId node_id,
                                  SemIR::InstId cond_id) -> SemIR::InstBlockId;

// Handles recovergence of control flow. Adds branches from the top
// `num_blocks` on the instruction block stack to a new block, pops the
// existing blocks, pushes the new block onto the instruction block stack,
// and adds it to the most recently pushed region.
auto AddConvergenceBlockAndPush(Context& context, Parse::NodeId node_id,
                                int num_blocks) -> void;

// Handles recovergence of control flow with a result value. Adds branches
// from the top few blocks on the instruction block stack to a new block, pops
// the existing blocks,  pushes the new block onto the instruction block
// stack, and adds it to the most recently pushed region. The number of blocks
// popped is the size of `block_args`, and the corresponding result values are
// the elements of `block_args`. Returns an instruction referring to the
// result value.
auto AddConvergenceBlockWithArgAndPush(
    Context& context, Parse::NodeId node_id,
    std::initializer_list<SemIR::InstId> block_args) -> SemIR::InstId;

// Sets the constant value of a block argument created as the result of a
// branch.  `select_id` should be a `BlockArg` that selects between two
// values. `cond_id` is the condition, `if_false` is the value to use if the
// condition is false, and `if_true` is the value to use if the condition is
// true.  We don't track enough information in the `BlockArg` inst for
// `TryEvalInst` to do this itself.
auto SetBlockArgResultBeforeConstantUse(Context& context,
                                        SemIR::InstId select_id,
                                        SemIR::InstId cond_id,
                                        SemIR::InstId if_true,
                                        SemIR::InstId if_false) -> void;

// Returns whether the current position in the current block is reachable.
auto IsCurrentPositionReachable(Context& context) -> bool;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CONTROL_FLOW_H_
