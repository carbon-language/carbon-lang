// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_OVERLOAD_RESOLUTION_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_OVERLOAD_RESOLUTION_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Checks whether a selected overload is accessible and diagnoses if not.
// `parent_scope_id`, if specified, describes the scope that was named to find
// the overload. If unspecified, we assume the overload was found in the class
// that it is a direct member of, rather than a derived class.
auto CheckCppOverloadAccess(
    Context& context, SemIR::LocId loc_id, clang::DeclAccessPair overload,
    SemIR::KnownInstId<SemIR::FunctionDecl> overload_inst_id,
    SemIR::NameScopeId parent_scope_id = SemIR::NameScopeId::None) -> void;

// Resolves which function to call using Clang overloading resolution, or
// returns an error instruction if overload resolution failed.
//
// A set with a single non-templated function goes through the same rules for
// overloading resolution. This is to make sure that calls that have no viable
// implicit conversion sequence are rejected even when an implicit conversion is
// possible. Keeping the same behavior here for consistency and supporting
// migrations so that the migrated callers from C++ remain valid.
auto PerformCppOverloadResolution(Context& context, SemIR::LocId loc_id,
                                  SemIR::CppOverloadSetId overload_set_id,
                                  SemIR::InstId self_id,
                                  llvm::ArrayRef<SemIR::InstId> arg_ids)
    -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_OVERLOAD_RESOLUTION_H_
