// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_MERGE_H_
#define CARBON_TOOLCHAIN_CHECK_MERGE_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Merges an import ref at new_inst_id another at prev_inst_id. May print a
// diagnostic if merging is invalid.
auto MergeImportRef(Context& context, SemIR::InstId new_inst_id,
                    SemIR::InstId prev_inst_id) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_MERGE_H_
