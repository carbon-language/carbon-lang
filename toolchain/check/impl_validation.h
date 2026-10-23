// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_IMPL_VALIDATION_H_
#define CARBON_TOOLCHAIN_CHECK_IMPL_VALIDATION_H_

#include "toolchain/check/context.h"

namespace Carbon::Check {

// Called after typechecking the full file to diagnose any `impl` declarations
// that are invalid because they are in wrong file or overlap with other `impl`
// declarations incorrectly.
auto ValidateImplsInFile(Context& context) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_IMPL_VALIDATION_H_
