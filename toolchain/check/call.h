// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CALL_H_
#define CARBON_TOOLCHAIN_CHECK_CALL_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Checks and builds SemIR for a call to `callee_id` with arguments `args_id`,
// where the callee is a function. `is_desugared` indicates that this call
// was produced by desugaring, not written as a function call in user code, so
// arguments to `ref` parameters aren't required to have `ref` tags.
auto PerformCallToFunction(Context& context, SemIR::LocId loc_id,
                           SemIR::InstId callee_id,
                           const SemIR::CalleeFunction& callee_function,
                           llvm::ArrayRef<SemIR::InstId> arg_ids,
                           bool is_desugared) -> SemIR::InstId;

// Checks and builds SemIR for a call to `callee_id` with arguments `args_id`.
// `is_desugared` indicates that this call
// was generated from an operator rather than from function call syntax, so
// arguments to `ref` parameters aren't required to have `ref` tags.
auto PerformCall(Context& context, SemIR::LocId loc_id, SemIR::InstId callee_id,
                 llvm::ArrayRef<SemIR::InstId> arg_ids,
                 bool is_desugared = false) -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CALL_H_
