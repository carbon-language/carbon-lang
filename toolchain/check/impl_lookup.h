// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_IMPL_LOOKUP_H_
#define CARBON_TOOLCHAIN_CHECK_IMPL_LOOKUP_H_

#include <variant>

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Looks up the witnesses to use for a type value or facet value, and a facet
// type naming a set of interfaces required to be implemented for that type, as
// well as possible constraints on those interfaces.
//
// N.B. In the future, `TypeType` will become a facet type, at which point type
// values will also be facet values.
//
// The return value is one of:
// - An InstBlockId value, containing an `ImplWitness` instruction for each
//   required interface in the `query_facet_type_const_id`. This verifies the
//   facet type is satisfied for the type in `type_const_id`, and provides a
//   witness for accessing the impl of each interface.
//
// - `InstBlockId::None`, indicating lookup failed for at least one required
//   interface in the `query_facet_type_const_id`. The facet type is not
//   satisfied for the type in `type_const_id`. This represents lookup failure,
//   but is not an error, so no diagnostic is emitted.
//
// - An error value, indicating the program is invalid and a diagonstic has been
//   produced, either in this function or before.
auto LookupImplWitness(Context& context, SemIR::LocId loc_id,
                       SemIR::ConstantId query_self_const_id,
                       SemIR::ConstantId query_facet_type_const_id)
    -> SemIR::InstBlockIdOrError;

struct UseOriginalImplSymbolicWitness {};
using LookupSingleImplWitnessResult =
    std::variant<SemIR::InstId, UseOriginalImplSymbolicWitness>;

// FIXME: Make a nicer return type that encapsulates variant and has
// .has_value() that includes being symbolic.

// FIXME: Receive const ImplSymbolicLookup& as a parameter below so only eval
// can call it.

// Looks for a witness instruction for a type value or facet value, and a single
// interface. This is for deferred lookup for an impl declaration. It does not
// consider the self facet value for finding a witness, since
// LookupImplWitness() would have found that and not caused us to defer lookup
// to here.
auto DeferredLookupSingleImplWitness(
    Context& context, SemIR::LocId loc_id,
    SemIR::ImplSymbolicWitness deferred_query)
    -> LookupSingleImplWitnessResult;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_IMPL_LOOKUP_H_
