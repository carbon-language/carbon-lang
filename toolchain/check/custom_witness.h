// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CUSTOM_WITNESS_H_
#define CARBON_TOOLCHAIN_CHECK_CUSTOM_WITNESS_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Builds a witness that the given type implements the given interface,
// populating it with the specified set of values. Returns a corresponding
// lookup result. Produces a diagnostic and returns `None` if the specified
// values aren't suitable for the interface.
auto BuildCustomWitness(Context& context, SemIR::LocId loc_id,
                        SemIR::ConstantId query_self_const_id,
                        SemIR::SpecificInterfaceId query_specific_interface_id,
                        llvm::ArrayRef<SemIR::InstId> values) -> SemIR::InstId;

// Builds a witness that the given type is copyable via a primitive copy.
auto BuildPrimitiveCopyWitness(
    Context& context, SemIR::LocId loc_id, SemIR::NameScopeId parent_scope_id,
    SemIR::ConstantId query_self_const_id,
    SemIR::SpecificInterfaceId query_specific_interface_id) -> SemIR::InstId;

// Builds a witness that the given type is trivially destroyable.
auto BuildTrivialDestroyWitness(
    Context& context, SemIR::LocId loc_id,
    SemIR::ConstantId query_self_const_id,
    SemIR::SpecificInterfaceId query_specific_interface_id) -> SemIR::InstId;

// Given an interface, returns the corresponding enum if it's covered by
// `CoreInterface`, or `Unknown` if it's some other interface.
auto GetCoreInterface(Context& context, SemIR::InterfaceId interface_id)
    -> SemIR::CoreInterface;

// Maps a `CoreInterface` to its `CoreIdentifier` equivalent.
auto AsCoreIdentifier(SemIR::CoreInterface core_interface) -> CoreIdentifier;

// Returns a witness for a `CoreInterface` `CustomWitness`. A return value of
// `None` indicates a non-final witness should be produced, while `std::nullopt`
// indicates the query is final and no witness can be produced.
//
// If `build_witness` is false, this function always returns `None` as the
// witness, whether it would be final or not. It is used to indicate the
// presence of such a witness without adding instructions for it.
auto LookupCustomWitness(Context& context, SemIR::LocId loc_id,
                         SemIR::CoreInterface core_interface,
                         SemIR::ConstantId query_self_const_id,
                         SemIR::SpecificInterfaceId query_specific_interface_id,
                         bool build_witness) -> std::optional<SemIR::InstId>;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CUSTOM_WITNESS_H_
