// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CALL_H_
#define CARBON_TOOLCHAIN_CHECK_CALL_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Checks and builds SemIR for a call to `callee_id` with arguments `args_id`.
auto PerformCall(Context& context, Parse::NodeId node_id,
                 SemIR::InstId callee_id, llvm::ArrayRef<SemIR::InstId> arg_ids)
    -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CALL_H_
