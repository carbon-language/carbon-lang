// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_INITIALIZING_EXPRESSION_H_
#define CARBON_TOOLCHAIN_CHECK_INITIALIZING_EXPRESSION_H_

#include "toolchain/check/context.h"
#include "toolchain/check/pending_block.h"
#include "toolchain/sem_ir/node.h"

namespace Carbon::Check {

// Materialize a temporary to hold the result of the given expression if it is
// an initializing expression.
auto MaterializeIfInitializing(Context& context, SemIR::NodeId expr_id)
    -> SemIR::NodeId;

// Commits to using a temporary to store the result of the initializing
// expression described by `init_id`, and returns the location of the
// temporary. If `discarded` is `true`, the result is discarded, and no
// temporary will be created if possible; if no temporary is created, the
// return value will be `SemIR::NodeId::Invalid`.
auto FinalizeTemporary(Context& context, SemIR::NodeId init_id, bool discarded)
    -> SemIR::NodeId;

// Marks the initializer `init_id` as initializing `target_id`.
auto MarkInitializerFor(Context& context, SemIR::NodeId init_id,
                        SemIR::NodeId target_id, PendingBlock& target_block)
    -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_INITIALIZING_EXPRESSION_H_
