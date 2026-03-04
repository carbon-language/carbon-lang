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

// Significant interfaces in `Core` which correspond to language features and
// can have custom witnesses.
enum class CoreInterface : std::int8_t {
  Copy = 1 << 0,
  Destroy = 1 << 1,

  Unknown = -1,
};

// Given an interface, returns the corresponding enum if it's covered by
// `CoreInterface`, or `Unknown` if it's some other interface.
auto GetCoreInterface(Context& context, SemIR::InterfaceId interface_id)
    -> CoreInterface;

// Returns a witness for a `CoreInterface` `CustomWitness`. A return value of
// `None` indicates a non-final witness should be produced, while `std::nullopt`
// indicates the query is final and no witness can be produced.
auto LookupCustomWitness(Context& context, SemIR::LocId loc_id,
                         CoreInterface core_interface,
                         SemIR::ConstantId query_self_const_id,
                         SemIR::SpecificInterfaceId query_specific_interface_id)
    -> std::optional<SemIR::InstId>;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CUSTOM_WITNESS_H_
