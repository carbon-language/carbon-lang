// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_PATTERN_MATCH_H_
#define CARBON_TOOLCHAIN_CHECK_PATTERN_MATCH_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// FIXME document how this differs from inputs
struct ParameterBlocks {
  // The implicit parameter list.
  SemIR::InstBlockId implicit_params_id;

  // The explicit parameter list.
  SemIR::InstBlockId params_id;
};

auto ProcessSignature(Context& context,
                      SemIR::InstBlockId implicit_param_patterns_id,
                      SemIR::InstBlockId param_patterns_id) -> ParameterBlocks;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_PATTERN_MATCH_H_
