// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_IMPL_LOOKUP_H_
#define CARBON_TOOLCHAIN_CHECK_IMPL_LOOKUP_H_

#include <variant>

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
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
                       SemIR::ConstantId query_facet_type_const_id,
                       bool diagnose = true) -> SemIR::InstBlockIdOrError;

// Returns whether the query matches against the given impl. This is like a
// `LookupImplWitness` operation but for a single interface, and against only
// the single impl.
auto LookupMatchesImpl(Context& context, SemIR::LocId loc_id,
                       SemIR::ConstantId query_self_const_id,
                       SemIR::SpecificInterface query_specific_interface,
                       SemIR::ImplId target_impl) -> bool;

// The kind of impl lookup being performed by a call to
// `EvalLookupSingleFinalWitness`.
enum class EvalImplLookupMode {
  // This is a regular impl lookup performed during check. If we produce a final
  // witness value that uses a specializable impl, the query will be poisoned so
  // that we will recheck it at the end of the compilation.
  Normal,
  // This is a re-check of a poisoned lookup being performed at the end of a
  // file. This disables any caching of lookup results for this query and redoes
  // the impl lookup.
  RecheckPoisonedLookup,
};

// Looks for a final witness for an impl lookup query consisting of a self (type
// or facet) and a single interface. This is for eval to execute lookup via the
// `LookupImplWitness` instruction. Since this query is re-evaluated against
// specifics, it provides monomorphization of the impl lookup, which allows for
// finding specializations.
auto EvalLookupSingleFinalWitness(Context& context, SemIR::LocId loc_id,
                                  SemIR::LookupImplWitness eval_query,
                                  SemIR::InstId self_facet_value_inst_id,
                                  EvalImplLookupMode mode) -> SemIR::ConstantId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_IMPL_LOOKUP_H_
