// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_OVERLOAD_RESOLUTION_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_OVERLOAD_RESOLUTION_H_

#include "toolchain/check/context.h"

namespace Carbon::Check {

// Performs overloading resolution for a call to an overloaded C++ set. A set
// with a single non-templated function is still considered to be an overload
// set, and goes through the same rules for checking the viability of the
// function. Uses Clang to find the best viable function for the call. Returns
// the resolved function, or `nullopt` if overload resolution failed.
auto PerformCppOverloadResolution(Context& context, SemIR::LocId loc_id,
                                  SemIR::InstId callee_id,
                                  llvm::ArrayRef<SemIR::InstId> arg_ids)
    -> std::optional<SemIR::InstId>;
}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_OVERLOAD_RESOLUTION_H_
