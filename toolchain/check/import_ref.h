// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_IMPORT_REF_H_
#define CARBON_TOOLCHAIN_CHECK_IMPORT_REF_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::Check {

// If the passed in instruction ID is an ImportRefUnloaded, loads it for use.
// The result will be an ImportRefUsed.
auto LoadImportRef(Context& context, SemIR::InstId inst_id, SemIRLoc loc)
    -> void;

// Load all impls declared in IRs imported into this context.
auto ImportImpls(Context& context) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_IMPORT_REF_H_
