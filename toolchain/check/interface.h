// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_INTERFACE_H_
#define CARBON_TOOLCHAIN_CHECK_INTERFACE_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Builds and returns an associated entity for `interface_id` corresponding to
// the declaration `decl_id`, which can be an associated function or an
// associated constant. Registers the associated entity in the list for the
// interface.
auto BuildAssociatedEntity(Context& context, SemIR::InterfaceId interface_id,
                           SemIR::InstId decl_id) -> SemIR::InstId;

// Gets the self specific of a generic declaration that is an interface member,
// given a specific for the interface plus a type to use as `Self`.
auto GetSelfSpecificForInterfaceMemberWithSelfType(
    Context& context, SemIRLoc loc, SemIR::SpecificId interface_specific_id,
    SemIR::GenericId generic_id, SemIR::SpecificId enclosing_specific_id,
    SemIR::TypeId self_type_id, SemIR::InstId witness_inst_id)
    -> SemIR::SpecificId;

// Gets the type of the specified associated entity, given the specific for the
// interface and the type of `Self`.
auto GetTypeForSpecificAssociatedEntity(Context& context, SemIRLoc loc,
                                        SemIR::SpecificId interface_specific_id,
                                        SemIR::InstId decl_id,
                                        SemIR::TypeId self_type_id,
                                        SemIR::InstId self_witness_id)
    -> SemIR::TypeId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_INTERFACE_H_
