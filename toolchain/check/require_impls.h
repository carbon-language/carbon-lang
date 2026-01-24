// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_REQUIRE_IMPLS_H_
#define CARBON_TOOLCHAIN_CHECK_REQUIRE_IMPLS_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/require_impls.h"

namespace Carbon::Check {

// A type-safe wrapper around the specific id of a RequireImpls entity.
// Constructed from a specific for its enclosing interface or constraint entity.
struct RequireImplsSpecific {
  SemIR::SpecificId specific_id;
};

// Get the specific of a RequireImpls from the specific of its enclosing
// interface or named constraint. Since a `require` declaration can not
// introduce new generic bindings, the specific for the RequireImpls can be
// constructed from the enclosing one.
auto GetRequireImplsSpecificFromEnclosingSpecific(
    Context& context, const SemIR::RequireImpls& require,
    SemIR::SpecificId enclosing_specific_id) -> RequireImplsSpecific;

// Like GetRequireImplsSpecificFromEnclosingSpecific but the `Self` value in the
// specific is replaced by a given facet value.
auto GetRequireImplsSpecificFromEnclosingSpecificWithSelfFacetValue(
    Context& context, const SemIR::RequireImpls& require,
    SemIR::SpecificId enclosing_specific_id,
    SemIR::ConstantId self_facet_value_id) -> RequireImplsSpecific;

// Returns the constant value of `inst_id` from inside a RequireImpls
// declaration, mapped into `enclosing_specific_id`. If an error results, it
// returns ErrorInst.
//
// An error can occur during monomorphization when the self-type was valid as a
// symbolic but becomes invalid with a more concrete specific.
//
// RequireImpls is always generic, so the instructions inside it must have a
// specific applied. That specific is constructed from a specific for the
// enclosing generic entity, with GetRequireImplsSpecificFromEnclosingSpecific.
auto GetConstantValueInRequireImplsSpecific(Context& context,
                                            RequireImplsSpecific specific,
                                            SemIR::InstId inst_id)
    -> SemIR::ConstantId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_REQUIRE_IMPLS_H_
