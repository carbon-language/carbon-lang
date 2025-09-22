// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_LOCATION_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_LOCATION_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Maps a Carbon source location into an equivalent Clang source location.
auto GetCppLocation(Context& context, SemIR::LocId loc_id)
    -> clang::SourceLocation;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_LOCATION_H_
