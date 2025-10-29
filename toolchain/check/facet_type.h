// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_FACET_TYPE_H_
#define CARBON_TOOLCHAIN_CHECK_FACET_TYPE_H_

#include <compare>

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Create a FacetType typed instruction object consisting of a interface. The
// `specific_id` specifies arguments in the case the interface is generic.
//
// The resulting FacetType may contain multiple interfaces if the named
// interface contains `require` declarations.
auto FacetTypeFromInterface(Context& context, SemIR::InterfaceId interface_id,
                            SemIR::SpecificId specific_id) -> SemIR::FacetType;

// Create a FacetType typed instruction object consisting of a named constraint.
// The `specific_id` specifies arguments in the case the named constraint is
// generic.
auto FacetTypeFromNamedConstraint(Context& context,
                                  SemIR::NamedConstraintId named_constraint_id,
                                  SemIR::SpecificId specific_id)
    -> SemIR::FacetType;

// Given an ImplWitnessAccessSubstituted, returns the InstId of the
// ImplWitnessAccess. Otherwise, returns the input `inst_id` unchanged.
//
// This must be used when accessing the LHS of a rewrite constraint which has
// not yet been resolved in order to preserve which associated constant is being
// rewritten.
auto GetImplWitnessAccessWithoutSubstitution(Context& context,
                                             SemIR::InstId inst_id)
    -> SemIR::InstId;

// Creates a impl witness instruction for a facet type. The facet type is
// required to be complete if `is_definition` is true or the facet type has
// rewrites. Otherwise a placeholder witness is created, and
// `AllocateFacetTypeImplWitness` can be used at the `impl` definition.
//
// Adds and returns an `ImplWitness` instruction (created with location set to
// `witness_loc_id`) that shows "`Self` type" of type "facet type" (the value of
// the `facet_type_inst_id` instruction) implements interface
// `interface_to_witness`, which must be an interface required by "facet type"
// (as determined by `RequireIdentifiedFacetType`). This witness reflects the
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
auto InitialFacetTypeImplWitness(
    Context& context, SemIR::LocId witness_loc_id,
    SemIR::TypeInstId facet_type_inst_id, SemIR::TypeInstId self_type_inst_id,
    const SemIR::SpecificInterface& interface_to_witness,
    SemIR::SpecificId self_specific_id, bool is_definition) -> SemIR::InstId;

// Returns `true` if the facet type is complete. Otherwise issues a diagnostic
// and returns `false`.
auto RequireCompleteFacetTypeForImplDefinition(
    Context& context, SemIR::LocId loc_id, SemIR::TypeInstId facet_type_inst_id)
    -> bool;

// Replaces the placeholder created by `InitialFacetTypeImplWitness` with an
// empty witness table of the right size. Requires the interface designated by
// `interface_id` to be complete.
auto AllocateFacetTypeImplWitness(Context& context,
                                  SemIR::InterfaceId interface_id,
                                  SemIR::InstBlockId witness_id) -> void;

// Perform rewrite constraint resolution for a facet type. The rewrite
// constraints resolution is described here:
// https://docs.carbon-lang.dev/docs/design/generics/appendix-rewrite-constraints.html#rewrite-constraint-resolution
//
// This function:
// * Replaces the RHS of rewrite rules referring to `.Self` with the value
//   coming from other rewrite rules. For example in `.X = () and .Y = .X` the
//   result is `.X = () and .Y = ()`.
// * Discards duplicate assignments to the same associated constant, such as in
//   `.X = () and .X = ()` which becomes just `.X = ()`.
// * Diagnoses multiple assignments of different values to the same associated
//   constant such as `.X = () and .X = .Y`.
// * Diagnoses cycles between rewrite rules such as `.X = .Y and .Y = .X` or
//   even `.X = .X`.
//
// The rewrite constraints in `rewrites` are modified in place and may be
// reordered, with `ErrorInst` inserted when diagnosing errors.
//
// Returns false if resolve failed due to diagnosing an error. The resulting
// value of the facet type should be an error constant.
auto ResolveFacetTypeRewriteConstraints(
    Context& context, SemIR::LocId loc_id,
    llvm::SmallVector<SemIR::FacetTypeInfo::RewriteConstraint>& rewrites)
    -> bool;

// Introduce `.Self` as a symbolic binding into the current scope, and return
// the `SymbolicBinding` instruction.
//
// The `self_type_id` is either a facet type (as `FacetType`) or `type` (as
// `TypeType`).
auto MakePeriodSelfFacetValue(Context& context, SemIR::TypeId self_type_id)
    -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_FACET_TYPE_H_
