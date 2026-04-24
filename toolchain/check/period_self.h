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
auto MakePeriodSelfFacetValue(Context& context, SemIR::TypeId self_type_id)
    -> SemIR::InstId;

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
