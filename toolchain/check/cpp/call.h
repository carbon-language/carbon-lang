// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_CALL_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_CALL_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Checks and builds SemIR for a call to a C++ function in the given overload
// set with self `self_id` and arguments `arg_ids`.
//
// Chooses the best viable C++ function by performing Clang overloading
// resolution over the overload set.
//
// Preserves the given self, if set. If not set, and the function is a C++
// member operator, self will be set to the first argument, which in turn will
// be removed from the given args.
//
// A set with a single non-templated function goes through the same rules for
// overloading resolution. This is to make sure that calls that have no viable
// implicit conversion sequence are rejected even when an implicit conversion is
// possible. Keeping the same behavior here for consistency and supporting
// migrations so that the migrated callers from C++ remain valid.
auto PerformCallToCppFunction(Context& context, SemIR::LocId loc_id,
                              SemIR::CppOverloadSetId overload_set_id,
                              SemIR::InstId self_id,
                              llvm::ArrayRef<SemIR::InstId> arg_ids)
    -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_CALL_H_
