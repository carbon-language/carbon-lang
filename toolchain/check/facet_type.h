// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_FACET_TYPE_H_
#define CARBON_TOOLCHAIN_CHECK_FACET_TYPE_H_

#include <compare>

#include "toolchain/check/context.h"
#include "toolchain/check/subst.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Create a FacetType typed instruction object consisting of a interface. The
// `specific_id` specifies arguments in the case the interface is generic.
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
// The type of `.Self` must be a `FacetType`, so that it gets wrapped in
// `FacetAccessType` when used in a type position, such as in `U:! I(.Self)`.
// This allows substitution with other facet values without requiring an
// additional `FacetAccessType` to be inserted.
auto MakePeriodSelfFacetValue(Context& context, SemIR::TypeId self_type_id)
    -> SemIR::InstId;

// Get a FacetType instruction for an empty FacetType. This is the facet
// equivalent to TypeType.
//
// TODO: We vaguely plan to replace TypeType with this FacetType in the future,
// though that's a big change.
auto GetEmptyFacetType(Context& context) -> SemIR::TypeId;

// Make a facet value for a type value, which has an empty FacetType as its
// type. Returns a constant value, whose instruction payload is a FacetValue.
auto GetConstantFacetValueForType(Context& context,
                                  SemIR::TypeInstId type_inst_id)
    -> SemIR::ConstantId;

auto GetConstantFacetValueForTypeAndInterface(
    Context& context, SemIR::TypeInstId type_inst_id,
    SemIR::SpecificInterface specific_interface, SemIR::InstId witness_id)
    -> SemIR::ConstantId;

class SubstPeriodSelfCallbacks : public SubstInstCallbacks {
 public:
  explicit SubstPeriodSelfCallbacks(
      Context* context, SemIR::LocId loc_id,
      SemIR::ConstantId period_self_replacement_id);
  auto Subst(SemIR::InstId& inst_id) -> SubstResult override;
  auto Rebuild(SemIR::InstId orig_inst_id, SemIR::Inst new_inst)
      -> SemIR::InstId override;

  virtual auto ShouldReplace(bool /*implicit*/) -> bool { return true; }

  auto loc_id() const -> SemIR::LocId { return loc_id_; }
  auto period_self_replacement_id() const -> SemIR::ConstantId {
    return period_self_replacement_id_;
  }

 private:
  auto GetReplacement(SemIR::InstId period_self, bool implicit)
      -> SemIR::InstId;
  auto ConvertReplacement(SemIR::InstId replacement_self_inst_id,
                          SemIR::TypeId replacement_type_id,
                          SemIR::TypeId period_self_type_id) -> SemIR::InstId;

  SemIR::LocId loc_id_;
  SemIR::ConstantId period_self_replacement_id_;

  // The last output of GetReplacement().
  SemIR::InstId cached_replacement_id_ = SemIR::InstId::None;
  // The type of the last output of GetReplacement(). If the type of `.Self`
  // matches, we can reuse the `cached_replacement_id_`.
  SemIR::TypeId cached_replacement_type_id_ = SemIR::TypeId::None;
};

// Replace all `.Self` references in `const_id`. The `callbacks` specifies the
// facet to replace them with.
auto SubstPeriodSelf(Context& context, SubstPeriodSelfCallbacks& callbacks,
                     SemIR::ConstantId const_id) -> SemIR::ConstantId;

// Replace all `.Self` references in the specific of the interface or named
// constraint. The `callbacks` specifies the facet to replace them with.
auto SubstPeriodSelf(Context& context, SubstPeriodSelfCallbacks& callbacks,
                     SemIR::SpecificInterface interface)
    -> SemIR::SpecificInterface;
auto SubstPeriodSelf(Context& context, SubstPeriodSelfCallbacks& callbacks,
                     SemIR::SpecificNamedConstraint constraint)
    -> SemIR::SpecificNamedConstraint;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_FACET_TYPE_H_
