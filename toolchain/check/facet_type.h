// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_FACET_TYPE_H_
#define CARBON_TOOLCHAIN_CHECK_FACET_TYPE_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Create a FacetType typed instruction object consisting of a single
// interface. The `specific_id` specifies arguments in the case the interface is
// generic.
auto FacetTypeFromInterface(Context& context, SemIR::InterfaceId interface_id,
                            SemIR::SpecificId specific_id) -> SemIR::FacetType;

// Adds and returns an `ImplWitness` instruction (created with location set to
// `witness_loc_id`) that shows "`Self` type" of type "facet type" (the value of
// the `facet_type_inst_id` instruction) implements interface
// `interface_to_witness`, which must be an interface required by "facet type"
// (as determined by `RequireCompleteFacetType`). This witness reflects the
// values assigned to associated constant members of that interface by rewrite
// constraints in the facet type. `self_specific_id` will be the `specific_id`
// of the resulting witness.
//
// `self_type_inst_id` is an instruction that evaluates to the `Self` type of
// the facet type. For example, in `T:! X where ...`, we will bind the `.Self`
// of the `where` facet type to `T`, and in `(X where ...) where ...`, we will
// bind the inner `.Self` to the outer `.Self`.
//
// If the facet type contains a rewrite, we may have deferred converting the
// rewritten value to the type of the associated constant. That conversion
// will also be performed as part of resolution, and may depend on the
// `Self` type.
auto ResolveFacetTypeImplWitness(
    Context& context, SemIR::LocId witness_loc_id,
    SemIR::InstId facet_type_inst_id, SemIR::InstId self_type_inst_id,
    const SemIR::SpecificInterface& interface_to_witness,
    SemIR::SpecificId self_specific_id) -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_FACET_TYPE_H_
