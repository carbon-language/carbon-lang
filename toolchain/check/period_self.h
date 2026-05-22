// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_PERIOD_SELF_H_
#define CARBON_TOOLCHAIN_CHECK_PERIOD_SELF_H_

#include "toolchain/check/context.h"
#include "toolchain/check/subst.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Introduce `.Self` as a symbolic binding into the current scope, and return
// the `SymbolicBinding` instruction.
//
// The type of `.Self` must be a `FacetType`, so that it gets wrapped in
// `FacetAccessType` when used in a type position, such as in `U:! I(.Self)`.
// This allows substitution with other facet values without requiring an
// additional `FacetAccessType` to be inserted.
auto MakePeriodSelfFacetValue(Context& context, SemIR::LocId loc_id,
                              SemIR::TypeId self_type_id) -> SemIR::InstId;

enum class SubstPeriodSelfBehaviour {
  ImplicitOnly,
  ExplicitOnly,
  All,
};

using SubstPeriodSelfRebuildInst =
    llvm::function_ref<auto(SemIR::Inst)->SemIR::InstId>;

// Replace `.Self` references in `const_id` with `period_self_replacement_id`.
//
// The `behaviour` specifies if all `.Self` are replaced or just implicit use in
// designators. The `rebuild` callback can optionally be specified to override
// how an instruction is re-constructed to form an InstId after replacement. It
// can return None to fall back to the default of evaluating the inst.
auto SubstPeriodSelf(
    Context& context, SemIR::LocId loc_id, SemIR::ConstantId const_id,
    SemIR::ConstantId period_self_replacement_id,
    SubstPeriodSelfBehaviour behaviour = SubstPeriodSelfBehaviour::All,
    SubstPeriodSelfRebuildInst rebuild = nullptr) -> SemIR::ConstantId;

// Replace `.Self` references in the specific of the interface or named
// constraint with `period_self_replacement_id`.
//
// The `behaviour` specifies if all `.Self` are replaced or just implicit use in
// designators. The `rebuild` callback can optionally be specified to override
// how an instruction is re-constructed to form an InstId after replacement. It
// can return None to fall back to the default of evaluating the inst.
auto SubstPeriodSelf(
    Context& context, SemIR::LocId loc_id, SemIR::SpecificInterface interface,
    SemIR::ConstantId period_self_replacement_id,
    SubstPeriodSelfBehaviour behaviour = SubstPeriodSelfBehaviour::All,
    SubstPeriodSelfRebuildInst rebuild = nullptr) -> SemIR::SpecificInterface;
auto SubstPeriodSelf(
    Context& context, SemIR::LocId loc_id,
    SemIR::SpecificNamedConstraint constraint,
    SemIR::ConstantId period_self_replacement_id,
    SubstPeriodSelfBehaviour behaviour = SubstPeriodSelfBehaviour::All,
    SubstPeriodSelfRebuildInst rebuild = nullptr)
    -> SemIR::SpecificNamedConstraint;

// Replace `.Self` references with the self-type. The `facet_type_inst_id` must
// be a `FacetType` instruction (or error).
//
// The implicit `.Self` in designators is not replaced in rewrite constraints,
// to allow for rewrite constraint resolution to recognise the designators.
// Later use of rewrite constraints requires further `.Self` replacement.
//
// Unlike SubstPeriodSelf, which works with constant values and thus canonical
// instructions, this operation can be done for non-canonical facet types. A new
// instruction is added for the output FacetType if anything does get replaced,
// and the original instruction id is preserved otherwise.
auto SubstPeriodSelfInFacetType(Context& context, SemIR::LocId loc_id,
                                SemIR::TypeInstId self_type_inst_id,
                                SemIR::TypeInstId facet_type_inst_id)
    -> SemIR::TypeInstId;

// Returns whether the constant value of `inst_id` is a reference to `.Self`.
//
// If `canonicalize` is true, look at the constant value of `inst_id` and get
// the canonicalized facet or type to look through FacetAccessType.
auto IsPeriodSelf(Context& context, SemIR::InstId inst_id,
                  bool canonicalize = true) -> bool;

// Look for ambiguous `.Self` in a `T impls X where ...` statement. The given
// inst ids are the non-canonical insts for the LHS and RHS of the `impls`
// inside a `where` expression.
//
// If the LHS is not `.Self` and RHS contains a nested `where` expression, the
// value of `.Self` becomes ambiguous on the RHS of the `where` (it could mean
// either the original value or new value given by the LHS of the `impls`). Note
// that implicit `.Self` references are never ambiguous, they always refer to
// the innermost value that `.Self` could refer to.
//
// Returns true if an error was diagnosed.
auto FindAndDiagnoseAmbiguousPeriodSelf(Context& context,
                                        SemIR::InstId impls_lhs_id,
                                        SemIR::InstId impls_rhs_id) -> bool;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_PERIOD_SELF_H_
