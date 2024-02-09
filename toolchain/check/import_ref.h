// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_IMPORT_REF_H_
#define CARBON_TOOLCHAIN_CHECK_IMPORT_REF_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::Check {

// If the passed in instruction ID is a ImportRefUnused, resolves it for use.
// Otherwise, errors.
auto TryResolveImportRefUnused(Context& context, SemIR::InstId inst_id) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_IMPORT_REF_H_
