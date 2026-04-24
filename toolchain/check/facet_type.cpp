// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/facet_type.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/control_flow.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/interface.h"
#include "toolchain/check/subst.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto FacetTypeFromInterface(Context& context, SemIR::InterfaceId interface_id,
                            SemIR::SpecificId specific_id) -> SemIR::FacetType {
  auto info = SemIR::FacetTypeInfo{};
  info.extend_constraints.push_back({interface_id, specific_id});
  info.Canonicalize();
  SemIR::FacetTypeId facet_type_id = context.facet_types().Add(info);
  return {.type_id = SemIR::TypeType::TypeId, .facet_type_id = facet_type_id};
}

auto FacetTypeFromNamedConstraint(Context& context,
                                  SemIR::NamedConstraintId named_constraint_id,
                                  SemIR::SpecificId specific_id)
    -> SemIR::FacetType {
  auto info = SemIR::FacetTypeInfo{};
  info.extend_named_constraints.push_back({named_constraint_id, specific_id});
  info.Canonicalize();
  SemIR::FacetTypeId facet_type_id = context.facet_types().Add(info);
  return {.type_id = SemIR::TypeType::TypeId, .facet_type_id = facet_type_id};
}

auto GetImplWitnessAccessWithoutSubstitution(Context& context,
                                             SemIR::InstId inst_id)
    -> SemIR::InstId {
  if (auto inst = context.insts().TryGetAs<SemIR::ImplWitnessAccessSubstituted>(
          inst_id)) {
    return inst->impl_witness_access_id;
  }
  return inst_id;
}

// A mapping of each associated constant (represented as `ImplWitnessAccess`) to
// its value (represented as an `InstId`). Used to track rewrite constraints,
// with the LHS mapping to the resolved value of the RHS.
class AccessRewriteValues {
 public:
  enum State {
    NotRewritten,
    BeingRewritten,
    FullyRewritten,
  };
  struct Value {
    State state;
    SemIR::InstId inst_id;
  };

  auto InsertNotRewritten(
      Context& context, SemIR::KnownInstId<SemIR::ImplWitnessAccess> access_id,
      SemIR::InstId inst_id) -> void {
    map_.Insert(context.constant_values().Get(access_id),
                {NotRewritten, inst_id});
  }

  // Finds and returns a pointer into the cache for a given ImplWitnessAccess.
  // The pointer will be invalidated by mutating the cache. Returns `nullptr`
  // if `access` is not found.
  auto FindRef(Context& context,
               SemIR::KnownInstId<SemIR::ImplWitnessAccess> access_id)
      -> Value* {
    auto result = map_.Lookup(context.constant_values().Get(access_id));
    if (!result) {
      return nullptr;
    }
    return &result.value();
  }

  auto SetBeingRewritten(Value& value) -> void {
    if (value.state == NotRewritten) {
      value.state = BeingRewritten;
    }
  }

  auto SetFullyRewritten(Context& context, Value& value, SemIR::InstId inst_id)
      -> void {
    if (value.state == FullyRewritten) {
      CARBON_CHECK(context.constant_values().Get(value.inst_id) ==
                   context.constant_values().Get(inst_id));
    }
    value = {FullyRewritten, inst_id};
  }

 private:
  // Try avoid heap allocations in the common case where there are a small
  // number of rewrite rules referring to each other by keeping up to 16 on
  // the stack.
  //
  // TODO: Revisit if 16 is an appropriate number when we can measure how deep
  // rewrite constraint chains go in practice.
  Map<SemIR::ConstantId, Value, 16> map_;
};

// To be used for substituting into the RHS of a rewrite constraint.
//
// It will substitute any `ImplWitnessAccess` into `.Self` (a reference to an
// associated constant) with the RHS of another rewrite constraint that writes
// to the same associated constant. For example:
// ```
// Z where .X = () and .Y = .X
// ```
// Here the second `.X` is an `ImplWitnessAccess` which would be substituted by
// finding the first rewrite constraint, where the LHS is for the same
// associated constant and using its RHS. So the substitution would produce:
// ```
// Z where .X = () and .Y = ()
// ```
//
// This additionally diagnoses cycles when the `ImplWitnessAccess` is reading
// from the same rewrite constraint, and is thus assigning to the associated
// constant a value that refers to the same associated constant, such as with `Z
// where .X = C(.X)`. In the event of a cycle, the `ImplWitnessAccess` is
// replaced with `ErrorInst` so that further evaluation of the
// `ImplWitnessAccess` will not loop infinitely.
//
// The `rewrite_values` given to the constructor must be set up initially with
// each rewrite rule of an associated constant inserted with its unresolved
// value via `InsertNotRewritten`. Then for each rewrite rule of an associated
// constant, the LHS access should be set as being rewritten with its state
// changed to `BeingRewritten` in order to detect cycles before performing
// SubstInst. The result of SubstInst should be preserved afterward by changing
// the state and value for the LHS to `FullyRewritten` and the subst output
// instruction, respectively, to avoid duplicating work.
class SubstImplWitnessAccessCallbacks : public SubstInstCallbacks {
 public:
  explicit SubstImplWitnessAccessCallbacks(Context* context,
                                           SemIR::LocId loc_id,
                                           AccessRewriteValues* rewrite_values)
      : SubstInstCallbacks(context),
        loc_id_(loc_id),
        rewrite_values_(rewrite_values) {}

  auto Subst(SemIR::InstId& rhs_inst_id) -> SubstResult override {
    auto rhs_access =
        context().insts().TryGetAsWithId<SemIR::ImplWitnessAccess>(rhs_inst_id);
    if (!rhs_access) {
      // We only want to substitute ImplWitnessAccesses written directly on the
      // RHS of the rewrite constraint, not when they are nested inside facet
      // types that are part of the RHS, like `.X = C as (I where .Y = {})`.
      if (context().insts().Is<SemIR::FacetType>(rhs_inst_id)) {
        return SubstResult::FullySubstituted;
      }
      if (context().constant_values().Get(rhs_inst_id).is_concrete()) {
        // There's no ImplWitnessAccess that we care about inside this
        // instruction.
        return SubstResult::FullySubstituted;
      }
      if (auto subst =
              context().insts().TryGetAs<SemIR::ImplWitnessAccessSubstituted>(
                  rhs_inst_id)) {
        // The reference to an associated constant was eagerly replaced with the
        // value of an earlier rewrite constraint, but may need further
        // substitution if it contains an `ImplWitnessAccess`.
        rhs_inst_id = subst->value_id;
        substs_in_progress_.push_back(rhs_inst_id);
        return SubstResult::SubstAgain;
      }
      // SubstOperands will result in a Rebuild or ReuseUnchanged callback, so
      // push the non-ImplWitnessAccess to get proper bracketing, allowing us
      // to pop it in the paired callback.
      substs_in_progress_.push_back(rhs_inst_id);
      return SubstResult::SubstOperands;
    }

    // If the access is going through a nested `ImplWitnessAccess`, that
    // access needs to be resolved to a facet value first. If it can't be
    // resolved then the outer one can not be either.
    if (auto lookup = context().insts().TryGetAs<SemIR::LookupImplWitness>(
            rhs_access->inst.witness_id)) {
      if (context().insts().Is<SemIR::ImplWitnessAccess>(
              lookup->query_self_inst_id)) {
        substs_in_progress_.push_back(rhs_inst_id);
        return SubstResult::SubstOperandsAndRetry;
      }
    }

    auto* rewrite_value =
        rewrite_values_->FindRef(context(), rhs_access->inst_id);
    if (!rewrite_value) {
      // The RHS refers to an associated constant for which there is no rewrite
      // rule.
      return SubstResult::FullySubstituted;
    }

    // Diagnose a cycle if the RHS refers to something that depends on the value
    // of the RHS.
    if (rewrite_value->state == AccessRewriteValues::BeingRewritten) {
      CARBON_DIAGNOSTIC(FacetTypeConstraintCycle, Error,
                        "found cycle in facet type constraint for {0}",
                        InstIdAsConstant);
      // TODO: It would be nice to note the places where the values are
      // assigned but rewrite constraint instructions are from canonical
      // constant values, and have no locations. We'd need to store a location
      // along with them in the rewrite constraints, and track propagation of
      // locations here, which may imply heap allocations.
      context().emitter().Emit(loc_id_, FacetTypeConstraintCycle, rhs_inst_id);
      rhs_inst_id = SemIR::ErrorInst::InstId;
      return SubstResult::FullySubstituted;
    } else if (rewrite_value->state == AccessRewriteValues::FullyRewritten) {
      rhs_inst_id = rewrite_value->inst_id;
      return SubstResult::FullySubstituted;
    }

    // We have a non-rewritten RHS. We need to recurse on rewriting it. Reuse
    // the previous lookup by mutating it in place.
    rewrite_values_->SetBeingRewritten(*rewrite_value);

    // The ImplWitnessAccess was replaced with some other instruction, which may
    // contain or be another ImplWitnessAccess. Keep track of the associated
    // constant we are now computing the value of.
    substs_in_progress_.push_back(rhs_inst_id);
    rhs_inst_id = rewrite_value->inst_id;
    return SubstResult::SubstAgain;
  }

  auto Rebuild(SemIR::InstId /*orig_inst_id*/, SemIR::Inst new_inst)
      -> SemIR::InstId override {
    auto inst_id = RebuildNewInst(loc_id_, new_inst);
    auto subst_inst_id = substs_in_progress_.pop_back_val();
    if (auto access =
            context().insts().TryGetAsWithId<SemIR::ImplWitnessAccess>(
                subst_inst_id)) {
      if (auto* rewrite_value =
              rewrite_values_->FindRef(context(), access->inst_id)) {
        rewrite_values_->SetFullyRewritten(context(), *rewrite_value, inst_id);
      }
    }
    return inst_id;
  }

  auto ReuseUnchanged(SemIR::InstId orig_inst_id) -> SemIR::InstId override {
    auto subst_inst_id = substs_in_progress_.pop_back_val();
    if (auto access =
            context().insts().TryGetAsWithId<SemIR::ImplWitnessAccess>(
                subst_inst_id)) {
      if (auto* rewrite_value =
              rewrite_values_->FindRef(context(), access->inst_id)) {
        rewrite_values_->SetFullyRewritten(context(), *rewrite_value,
                                           orig_inst_id);
      }
    }
    return orig_inst_id;
  }

 private:
  struct SubstInProgress {
    // The associated constant whose value is being determined, represented as
    // an ImplWitnessAccess. Or another instruction that we are recursing
    // through.
    SemIR::InstId inst_id;
  };

  // The location of the rewrite constraints as a whole.
  SemIR::LocId loc_id_;
  // Tracks the resolved value of each rewrite constraint, keyed by the
  // `ImplWitnessAccess` of the associated constant on the LHS of the
  // constraint. The value of each associated constant may be changed during
  // substitution, replaced with a fully resolved value for the RHS. This allows
  // us to cache work; when a value for an associated constant is found once it
  // can be reused cheaply, avoiding exponential runtime when rewrite rules
  // refer to each other in ways that create exponential references.
  AccessRewriteValues* rewrite_values_;
  // A stack of instructions being replaced in Subst(). When it's an associated
  // constant, then it represents the constant value is being determined,
  // represented as an ImplWitnessAccess.
  //
  // Avoid heap allocations in common cases, if there are chains of instructions
  // in associated constants with a depth at most 16.
  llvm::SmallVector<SemIR::InstId, 16> substs_in_progress_;
};

auto ResolveFacetTypeRewriteConstraints(
    Context& context, SemIR::LocId loc_id,
    llvm::SmallVector<SemIR::FacetTypeInfo::RewriteConstraint>& rewrites)
    -> bool {
  if (rewrites.empty()) {
    return true;
  }

  AccessRewriteValues rewrite_values;

  for (auto& constraint : rewrites) {
    auto lhs_access = context.insts().TryGetAsWithId<SemIR::ImplWitnessAccess>(
        GetImplWitnessAccessWithoutSubstitution(context, constraint.lhs_id));
    if (!lhs_access) {
      continue;
    }

    rewrite_values.InsertNotRewritten(context, lhs_access->inst_id,
                                      constraint.rhs_id);
  }

  for (auto& constraint : rewrites) {
    auto lhs_access = context.insts().TryGetAsWithId<SemIR::ImplWitnessAccess>(
        GetImplWitnessAccessWithoutSubstitution(context, constraint.lhs_id));
    if (!lhs_access) {
      continue;
    }

    auto* lhs_rewrite_value =
        rewrite_values.FindRef(context, lhs_access->inst_id);
    // Every LHS was added with InsertNotRewritten above.
    CARBON_CHECK(lhs_rewrite_value);
    rewrite_values.SetBeingRewritten(*lhs_rewrite_value);

    auto replace_witness_callbacks =
        SubstImplWitnessAccessCallbacks(&context, loc_id, &rewrite_values);
    auto rhs_subst_inst_id =
        SubstInst(context, constraint.rhs_id, replace_witness_callbacks);
    if (rhs_subst_inst_id == SemIR::ErrorInst::InstId) {
      return false;
    }

    if (lhs_rewrite_value->state == AccessRewriteValues::FullyRewritten) {
      auto rhs_existing_const_id =
          context.constant_values().Get(lhs_rewrite_value->inst_id);
      auto rhs_subst_const_id =
          context.constant_values().Get(rhs_subst_inst_id);
      if (rhs_subst_const_id != rhs_existing_const_id) {
        if (rhs_existing_const_id != SemIR::ErrorInst::ConstantId) {
          CARBON_DIAGNOSTIC(AssociatedConstantWithDifferentValues, Error,
                            "associated constant {0} given two different "
                            "values {1} and {2}",
                            InstIdAsConstant, InstIdAsConstant,
                            InstIdAsConstant);
          // Use inst id ordering as a simple proxy for source ordering, to
          // try to name the values in the same order they appear in the facet
          // type.
          auto source_order1 =
              lhs_rewrite_value->inst_id.index < rhs_subst_inst_id.index
                  ? lhs_rewrite_value->inst_id
                  : rhs_subst_inst_id;
          auto source_order2 =
              lhs_rewrite_value->inst_id.index >= rhs_subst_inst_id.index
                  ? lhs_rewrite_value->inst_id
                  : rhs_subst_inst_id;
          // TODO: It would be nice to note the places where the values are
          // assigned but rewrite constraint instructions are from canonical
          // constant values, and have no locations. We'd need to store a
          // location along with them in the rewrite constraints.
          context.emitter().Emit(loc_id, AssociatedConstantWithDifferentValues,
                                 GetImplWitnessAccessWithoutSubstitution(
                                     context, constraint.lhs_id),
                                 source_order1, source_order2);
        }
        return false;
      }
    }

    rewrite_values.SetFullyRewritten(context, *lhs_rewrite_value,
                                     rhs_subst_inst_id);
  }

  // Rebuild the `rewrites` vector with resolved values for the RHS. Drop any
  // duplicate rewrites in the `rewrites` vector by walking through the
  // `rewrite_values` map and dropping the computed RHS value for each LHS the
  // first time we see it, and erasing the constraint from the vector if we see
  // the same LHS again.
  size_t keep_size = rewrites.size();
  for (size_t i = 0; i < keep_size;) {
    auto& constraint = rewrites[i];

    auto lhs_access = context.insts().TryGetAsWithId<SemIR::ImplWitnessAccess>(
        GetImplWitnessAccessWithoutSubstitution(context, constraint.lhs_id));
    if (!lhs_access) {
      ++i;
      continue;
    }

    auto& rewrite_value = *rewrite_values.FindRef(context, lhs_access->inst_id);
    auto rhs_id = std::exchange(rewrite_value.inst_id, SemIR::InstId::None);
    if (rhs_id == SemIR::InstId::None) {
      std::swap(rewrites[i], rewrites[keep_size - 1]);
      --keep_size;
    } else {
      rewrites[i].rhs_id = rhs_id;
      ++i;
    }
  }
  rewrites.erase(rewrites.begin() + keep_size, rewrites.end());

  return true;
}

auto MakePeriodSelfFacetValue(Context& context, SemIR::TypeId self_type_id)
    -> SemIR::InstId {
  CARBON_CHECK(self_type_id == SemIR::ErrorInst::TypeId ||
               context.types().Is<SemIR::FacetType>(self_type_id));
  auto entity_name_id = context.entity_names().AddCanonical({
      .name_id = SemIR::NameId::PeriodSelf,
      .parent_scope_id = context.scope_stack().PeekNameScopeId(),
  });
  auto inst_id = AddInst(
      context, SemIR::LocIdAndInst::NoLoc<SemIR::SymbolicBinding>({
                   .type_id = self_type_id,
                   .entity_name_id = entity_name_id,
                   // `None` because there is no equivalent non-symbolic value.
                   .value_id = SemIR::InstId::None,
               }));
  auto existing = context.scope_stack().LookupOrAddName(
      SemIR::NameId::PeriodSelf, inst_id, ScopeIndex::None,
      IsCurrentPositionReachable(context));
  // Shouldn't have any names in newly created scope.
  CARBON_CHECK(!existing.has_value());
  return inst_id;
}

auto GetEmptyFacetType(Context& context) -> SemIR::TypeId {
  SemIR::FacetTypeId facet_type_id =
      context.facet_types().Add(SemIR::FacetTypeInfo{});
  auto const_id = EvalOrAddInst<SemIR::FacetType>(
      context, SemIR::LocId::None,
      {.type_id = SemIR::TypeType::TypeId, .facet_type_id = facet_type_id});
  return context.types().GetTypeIdForTypeConstantId(const_id);
}

auto GetConstantFacetValueForType(Context& context,
                                  SemIR::TypeInstId type_inst_id)
    -> SemIR::ConstantId {
  // We use an empty facet type because values of type `type` do not provide any
  // witnesses of their own.
  auto type_facet_type = GetEmptyFacetType(context);
  return EvalOrAddInst<SemIR::FacetValue>(
      context, SemIR::LocId::None,
      {.type_id = type_facet_type,
       .type_inst_id = type_inst_id,
       .witnesses_block_id = SemIR::InstBlockId::Empty});
}

auto GetConstantFacetValueForTypeAndInterface(
    Context& context, SemIR::TypeInstId type_inst_id,
    SemIR::SpecificInterface specific_interface, SemIR::InstId witness_id)
    -> SemIR::ConstantId {
  // Get the type of the inner `Self`, which is the facet type of the interface.
  auto interface_facet_type = EvalOrAddInst(
      context, SemIR::LocId::None,
      FacetTypeFromInterface(context, specific_interface.interface_id,
                             specific_interface.specific_id));
  auto self_facet_type_in_generic_without_self =
      context.types().GetTypeIdForTypeConstantId(interface_facet_type);

  auto witnesses_block_id = context.inst_blocks().AddCanonical({witness_id});
  auto self_value_const_id = EvalOrAddInst<SemIR::FacetValue>(
      context, SemIR::LocId::None,
      {.type_id = self_facet_type_in_generic_without_self,
       .type_inst_id = type_inst_id,
       .witnesses_block_id = witnesses_block_id});
  return self_value_const_id;
}

SubstPeriodSelfCallbacks::SubstPeriodSelfCallbacks(
    Context* context, SemIR::LocId loc_id,
    SemIR::ConstantId period_self_replacement_id)
    : SubstInstCallbacks(context),
      loc_id_(loc_id),
      period_self_replacement_id_(period_self_replacement_id) {}

auto SubstPeriodSelfCallbacks::Subst(SemIR::InstId& inst_id) -> SubstResult {
  // FacetTypes are concrete even if they have `.Self` inside them, but we
  // don't recurse into FacetTypes, so we can use this as a base case. This
  // avoids infinite recursion on TypeType and ErrorInst.
  if (context().constant_values().Get(inst_id).is_concrete()) {
    return FullySubstituted;
  }
  // Don't recurse into nested facet types, even if they are symbolic. Leave
  // their `.Self` as is.
  if (context().insts().Is<SemIR::FacetType>(inst_id)) {
    return FullySubstituted;
  }

  // For `.X` (which is `.Self.X`) replace the `.Self` in the query self
  // position, and report it as `implicit`. Any `.Self` references in the
  // specific interface would be replaced later and not treated as `implicit`.
  //
  // TODO: This all goes away when eval doesn't need to know about implicit
  // .Self for diagnostics, once we diagnose invalid `.Self` in name lookup.
  if (auto access =
          context().insts().TryGetAs<SemIR::ImplWitnessAccess>(inst_id)) {
    if (auto witness = context().insts().TryGetAs<SemIR::LookupImplWitness>(
            access->witness_id)) {
      if (auto bind = context().insts().TryGetAs<SemIR::SymbolicBinding>(
              witness->query_self_inst_id)) {
        const auto& entity_name =
            context().entity_names().Get(bind->entity_name_id);
        if (entity_name.name_id == SemIR::NameId::PeriodSelf) {
          auto replacement_id =
              GetReplacement(witness->query_self_inst_id, true);
          auto new_witness =
              Rebuild(access->witness_id,
                      SemIR::LookupImplWitness{
                          .type_id = witness->type_id,
                          .query_self_inst_id = replacement_id,
                          // Don't replace `.Self` in the interface specific
                          // here. That is an explicit `.Self` use. We'll
                          // revisit the instruction for that.
                          .query_specific_interface_id =
                              witness->query_specific_interface_id,
                      });
          auto new_access = Rebuild(inst_id, SemIR::ImplWitnessAccess{
                                                 .type_id = access->type_id,
                                                 .witness_id = new_witness,
                                                 .index = access->index,
                                             });
          inst_id = new_access;
          return SubstAgain;
        }
      }
    }
  }

  if (auto bind_type =
          context().insts().TryGetAs<SemIR::SymbolicBindingType>(inst_id)) {
    auto bind = context().insts().GetAs<SemIR::SymbolicBinding>(
        bind_type->facet_value_inst_id);
    const auto& entity_name = context().entity_names().Get(bind.entity_name_id);
    if (entity_name.name_id == SemIR::NameId::PeriodSelf) {
      // We need to re-construct SymbolicBindingType, not just replace its
      // facet instruction. If the new operand is a SymbolicBinding, we need
      // to use its entity name.
      inst_id = Rebuild(
          inst_id,
          SemIR::FacetAccessType{.type_id = SemIR::TypeType::TypeId,
                                 .facet_value_inst_id = GetReplacement(
                                     bind_type->facet_value_inst_id, false)});
      return FullySubstituted;
    }
  }

  if (auto bind = context().insts().TryGetAs<SemIR::SymbolicBinding>(inst_id)) {
    const auto& entity_name =
        context().entity_names().Get(bind->entity_name_id);
    if (entity_name.name_id == SemIR::NameId::PeriodSelf) {
      inst_id = GetReplacement(inst_id, false);
      return FullySubstituted;
    }
  }

  return SubstOperands;
}

auto SubstPeriodSelfCallbacks::Rebuild(SemIR::InstId orig_inst_id,
                                       SemIR::Inst new_inst) -> SemIR::InstId {
  return RebuildNewInst(SemIR::LocId(orig_inst_id), new_inst);
}

auto SubstPeriodSelfCallbacks::GetReplacement(SemIR::InstId period_self,
                                              bool implicit) -> SemIR::InstId {
  if (!ShouldReplace(implicit)) {
    return period_self;
  }

  auto period_self_type_id = context().insts().Get(period_self).type_id();
  CARBON_CHECK(context().types().Is<SemIR::FacetType>(period_self_type_id));

  auto replacement_self_inst_id =
      context().constant_values().GetInstId(period_self_replacement_id_);
  auto replacement_type_id =
      context().insts().Get(replacement_self_inst_id).type_id();
  CARBON_CHECK(context().types().IsFacetType(replacement_type_id));

  // If the replacement has the same type as `.Self`, use it directly.
  if (replacement_type_id == period_self_type_id) {
    return replacement_self_inst_id;
  }

  // If we have already converted the replacement to the type of `.Self`, use
  // our previous conversion.
  if (period_self_type_id == cached_replacement_type_id_) {
    return cached_replacement_id_;
  }

  // Convert the replacement facet to the type of `.Self`.
  cached_replacement_id_ = ConvertReplacement(
      replacement_self_inst_id, replacement_type_id, period_self_type_id);
  cached_replacement_type_id_ = period_self_type_id;
  return cached_replacement_id_;
}

auto SubstPeriodSelfCallbacks::ConvertReplacement(
    SemIR::InstId replacement_self_inst_id, SemIR::TypeId replacement_type_id,
    SemIR::TypeId period_self_type_id) -> SemIR::InstId {
  // TODO: Replace all empty facet types with TypeType.
  if (period_self_type_id == GetEmptyFacetType(context())) {
    // Convert to an empty facet type (representing TypeType); we don't need
    // any witnesses.
    return ConvertToValueOfType(context(), loc_id_, replacement_self_inst_id,
                                period_self_type_id);
  }

  // We have a facet or a type, but we need more interfaces in the facet type.
  // We will have to synthesize a symbolic witness for each interface.
  //
  // Why is this okay? The type of `.Self` comes from interfaces that are
  // before it (to the left of it) in the facet type. The replacement for
  // `.Self` will have to impl those interfaces in order to match the facet
  // type, so we know that it is valid to construct these witnesses.

  // Make the replacement into a type, which we will need for the FacetValue.
  if (context().types().Is<SemIR::FacetType>(replacement_type_id)) {
    replacement_self_inst_id = context().constant_values().GetInstId(
        EvalOrAddInst<SemIR::FacetAccessType>(
            context(), loc_id_,
            {.type_id = SemIR::TypeType::TypeId,
             .facet_value_inst_id = replacement_self_inst_id}));
  }

  auto period_self_facet_type =
      context().types().GetAs<SemIR::FacetType>(period_self_type_id);
  auto identified_period_self_type_id = RequireIdentifiedFacetType(
      context(), loc_id_,
      context().constant_values().Get(replacement_self_inst_id),
      period_self_facet_type, [&](auto& /*builder*/) {
        // The facet type containing this `.Self` should have already been
        // identified, which would ensure that the type of `.Self` can be
        // identified since it can only depend on things to the left of it
        // inside the same facet type.
        CARBON_FATAL("could not identify type of `.Self`");
      });
  const auto& identified_period_self_type =
      context().identified_facet_types().Get(identified_period_self_type_id);
  auto required_impls = identified_period_self_type.required_impls();
  llvm::SmallVector<SemIR::InstId> witnesses;
  witnesses.reserve(required_impls.size());
  for (const auto& req : required_impls) {
    witnesses.push_back(context().constant_values().GetInstId(
        EvalOrAddInst<SemIR::LookupImplWitness>(
            context(), loc_id_,
            {.type_id =
                 GetSingletonType(context(), SemIR::WitnessType::TypeInstId),
             .query_self_inst_id =
                 context().constant_values().GetInstId(req.self_facet_value),
             .query_specific_interface_id = context().specific_interfaces().Add(
                 req.specific_interface)})));
  }
  return context().constant_values().GetInstId(EvalOrAddInst<SemIR::FacetValue>(
      context(), loc_id_,
      {
          .type_id = period_self_type_id,
          .type_inst_id =
              context().types().GetAsTypeInstId(replacement_self_inst_id),
          .witnesses_block_id = context().inst_blocks().Add(witnesses),
      }));
}

auto SubstPeriodSelf(Context& context, SubstPeriodSelfCallbacks& callbacks,
                     SemIR::ConstantId const_id) -> SemIR::ConstantId {
  // Don't replace `.Self` with itself; that is cyclical.
  //
  // If the types differ, we would try to convert the replacement to a `.Self`
  // of the desired type in `const_id`, which is what we already have, so
  // there's nothing we need to do. But trying to do that conversion recurses
  // when the type of the `.Self` contains a `.Self`.
  if (auto bind_type =
          context.constant_values().TryGetInstAs<SemIR::SymbolicBinding>(
              GetCanonicalFacetOrTypeValue(
                  context, callbacks.period_self_replacement_id()))) {
    if (context.entity_names().Get(bind_type->entity_name_id).name_id ==
        SemIR::NameId::PeriodSelf) {
      return const_id;
    }
  }

  auto subst_id = SubstInst(
      context, context.constant_values().GetInstId(const_id), callbacks);
  return context.constant_values().Get(subst_id);
}

static auto SubstPeriodSelfInSpecific(Context& context,
                                      SubstPeriodSelfCallbacks& callbacks,
                                      SemIR::SpecificId specific_id)
    -> SemIR::SpecificId {
  if (!specific_id.has_value()) {
    return specific_id;
  }

  const auto& specific = context.specifics().Get(specific_id);

  // Substitute into the specific without having to construct a FacetType
  // instruction just to hold the specific interface inside a constant id.
  llvm::SmallVector<SemIR::InstId> args(
      context.inst_blocks().Get(specific.args_id));
  for (auto& arg_id : args) {
    auto const_id = context.constant_values().Get(arg_id);
    const_id = SubstPeriodSelf(context, callbacks, const_id);
    arg_id = context.constant_values().GetInstId(const_id);
  }
  return MakeSpecific(context, callbacks.loc_id(), specific.generic_id, args);
}

auto SubstPeriodSelf(Context& context, SubstPeriodSelfCallbacks& callbacks,
                     SemIR::SpecificInterface interface)
    -> SemIR::SpecificInterface {
  interface.specific_id =
      SubstPeriodSelfInSpecific(context, callbacks, interface.specific_id);
  return interface;
}
auto SubstPeriodSelf(Context& context, SubstPeriodSelfCallbacks& callbacks,
                     SemIR::SpecificNamedConstraint constraint)
    -> SemIR::SpecificNamedConstraint {
  constraint.specific_id =
      SubstPeriodSelfInSpecific(context, callbacks, constraint.specific_id);
  return constraint;
}

}  // namespace Carbon::Check
