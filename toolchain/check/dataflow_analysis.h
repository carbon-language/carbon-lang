// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_DATAFLOW_ANALYSIS_H_
#define CARBON_TOOLCHAIN_CHECK_DATAFLOW_ANALYSIS_H_

#include "common/set.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/check/context.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::Check {

// Performs various dataflow analysis checks on the SemIR.
auto RunDataflowAnalysis(Context& context, SemIR::FunctionId function_id)
    -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_DATAFLOW_ANALYSIS_H_
