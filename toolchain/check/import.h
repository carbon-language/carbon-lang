// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_IMPORT_H_
#define CARBON_TOOLCHAIN_CHECK_IMPORT_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::Check {

// Add imports to the root block.
auto Import(Context& context, SemIR::TypeId namespace_type_id,
            const SemIR::File& import_sem_ir) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_IMPORT_H_
