// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/period_self.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/control_flow.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/facet_type.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/impl_lookup.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/subst.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto MakePeriodSelfFacetValue(Context& context, SemIR::LocId loc_id,
                              SemIR::TypeId self_type_id) -> SemIR::InstId {
  CARBON_CHECK(self_type_id == SemIR::ErrorInst::TypeId ||
               context.types().Is<SemIR::FacetType>(self_type_id));
  auto entity_name_id = context.entity_names().AddCanonical(
      {.name_id = SemIR::NameId::PeriodSelf,
       .parent_scope_id = context.scope_stack().PeekNameScopeId(),
       .is_frozen_period_self = true});
  auto inst_id = AddInst<SemIR::SymbolicBinding>(
      context, loc_id,
      {
          .type_id = self_type_id,
          .entity_name_id = entity_name_id,
          // `None` because there is no equivalent non-symbolic value.
          .value_id = SemIR::InstId::None,
      });
  auto existing = context.scope_stack().LookupOrAddName(
      SemIR::NameId::PeriodSelf, inst_id, ScopeIndex::None,
      IsCurrentPositionReachable(context));
  // Shouldn't have any names in newly created scope.
  CARBON_CHECK(!existing.has_value());
  return inst_id;
}

struct GetAsResult {
  SemIR::SymbolicBinding bind;
  bool is_frozen;
};

static auto TryGetAsPeriodSelf(Context& context, SemIR::InstId inst_id,
                               bool canonicalize)
    -> std::optional<GetAsResult> {
  auto const_inst_id = context.constant_values().GetConstantInstId(inst_id);
  if (!const_inst_id.has_value()) {
    return std::nullopt;
  }
  auto query_inst_id =
      canonicalize ? GetCanonicalFacetOrTypeValue(context, const_inst_id)
                   : inst_id;
  if (auto bind =
          context.insts().TryGetAs<SemIR::SymbolicBinding>(query_inst_id)) {
    const auto& entity_name = context.entity_names().Get(bind->entity_name_id);
    if (entity_name.name_id == SemIR::NameId::PeriodSelf) {
      return {{*bind, entity_name.is_frozen_period_self}};
    }
  }
  return std::nullopt;
}

class SubstPeriodSelfCallbacks : public SubstInstCallbacks {
 public:
  explicit SubstPeriodSelfCallbacks(
      Context* context, SemIR::LocId loc_id,
      SemIR::ConstantId period_self_replacement_id,
      SubstPeriodSelfRebuildInst rebuild)
      : SubstInstCallbacks(context),
        loc_id_(loc_id),
        period_self_replacement_id_(period_self_replacement_id),
        rebuild_callback_(rebuild) {}

  auto Subst(SemIR::InstId& inst_id) -> SubstResult override {
    // We need to recurse into facet types that are concrete to find `.Self`,
    // because the top level instruction being substituted could be such a facet
    // type. So we can't early out if the inst has a concrete constant value.
    if (inst_id == SemIR::TypeType::TypeInstId ||
        inst_id == SemIR::ErrorInst::InstId) {
      return FullySubstituted;
    }

    // Canonicalization not necessary; we are working with the constant
    // value already, and the query self in a witness is already
    // canonicalized.
    if (auto period_self = TryGetAsPeriodSelf(context(), inst_id,
                                              /*canonicalize=*/false)) {
      if (!period_self->is_frozen) {
        inst_id = GetReplacement(inst_id);
      }
      return FullySubstituted;
    }

    return SubstOperandsSkipType;
  }

  auto Rebuild(SemIR::InstId orig_inst_id, SemIR::Inst new_inst)
      -> SemIR::InstId override {
    if (rebuild_callback_) {
      if (auto inst_id = rebuild_callback_(new_inst); inst_id.has_value()) {
        return inst_id;
      }
    }
    return RebuildNewInst(SemIR::LocId(orig_inst_id), new_inst);
  }

 private:
  auto GetReplacement(SemIR::InstId period_self) -> SemIR::InstId {
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
    cached_replacement_id_ =
        ConvertReplacement(replacement_self_inst_id, replacement_type_id,
                           period_self, period_self_type_id);
    cached_replacement_type_id_ = period_self_type_id;
    return cached_replacement_id_;
  }

  auto ConvertReplacement(SemIR::InstId replacement_self_inst_id,
                          SemIR::TypeId replacement_type_id,
                          SemIR::InstId period_self_inst_id,
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

    auto witnesses = MakeWitnessesForPeriodSelfTypeWithoutLookup(
        context(), loc_id_,
        context().constant_values().Get(replacement_self_inst_id),
        context().constant_values().Get(period_self_inst_id));
    if (witnesses.has_error_value()) {
      return SemIR::ErrorInst::InstId;
    }
    return context().constant_values().GetInstId(
        EvalOrAddInst<SemIR::FacetValue>(
            context(), loc_id_,
            {
                .type_id = period_self_type_id,
                .type_inst_id =
                    context().types().GetAsTypeInstId(replacement_self_inst_id),
                .witnesses_block_id = witnesses.inst_block_id(),
            }));
  }

  SemIR::LocId loc_id_;
  SemIR::ConstantId period_self_replacement_id_;
  SubstPeriodSelfRebuildInst rebuild_callback_;

  // The last output of GetReplacement().
  SemIR::InstId cached_replacement_id_ = SemIR::InstId::None;
  // The type of the last output of GetReplacement(). If the type of `.Self`
  // matches, we can reuse the `cached_replacement_id_`.
  SemIR::TypeId cached_replacement_type_id_ = SemIR::TypeId::None;
};

auto SubstPeriodSelf(Context& context, SemIR::LocId loc_id,
                     SemIR::ConstantId const_id,
                     SemIR::ConstantId period_self_replacement_id,
                     SubstPeriodSelfRebuildInst rebuild) -> SemIR::ConstantId {
  // Don't replace `.Self` with itself; that is cyclical.
  //
  // If the types differ, we would try to convert the replacement to a `.Self`
  // of the desired type in `const_id`, which is what we already have, so
  // there's nothing we need to do. But trying to do that conversion recurses
  // when the type of the `.Self` contains a `.Self`.
  if (IsPeriodSelf(context, context.constant_values().GetInstId(
                                period_self_replacement_id))) {
    return const_id;
  }

  SubstPeriodSelfCallbacks callbacks(&context, loc_id,
                                     period_self_replacement_id, rebuild);
  auto subst_id = SubstInst(
      context, context.constant_values().GetInstId(const_id), callbacks);
  return context.constant_values().Get(subst_id);
}

static auto SubstPeriodSelfInSpecific(
    Context& context, SemIR::LocId loc_id, SemIR::SpecificId specific_id,
    SemIR::ConstantId period_self_replacement_id,
    SubstPeriodSelfRebuildInst rebuild) -> SemIR::SpecificId {
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
    const_id = SubstPeriodSelf(context, loc_id, const_id,
                               period_self_replacement_id, rebuild);
    arg_id = context.constant_values().GetInstId(const_id);
  }
  return MakeSpecific(context, loc_id, specific.generic_id, args);
}

auto SubstPeriodSelf(Context& context, SemIR::LocId loc_id,
                     SemIR::SpecificInterface interface,
                     SemIR::ConstantId period_self_replacement_id,
                     SubstPeriodSelfRebuildInst rebuild)
    -> SemIR::SpecificInterface {
  interface.specific_id =
      SubstPeriodSelfInSpecific(context, loc_id, interface.specific_id,
                                period_self_replacement_id, rebuild);
  return interface;
}
auto SubstPeriodSelf(Context& context, SemIR::LocId loc_id,
                     SemIR::SpecificNamedConstraint constraint,
                     SemIR::ConstantId period_self_replacement_id,
                     SubstPeriodSelfRebuildInst rebuild)
    -> SemIR::SpecificNamedConstraint {
  constraint.specific_id =
      SubstPeriodSelfInSpecific(context, loc_id, constraint.specific_id,
                                period_self_replacement_id, rebuild);
  return constraint;
}

auto SubstPeriodSelfInFacetType(Context& context, SemIR::LocId loc_id,
                                SemIR::InstId self_inst_id,
                                SemIR::TypeInstId facet_type_inst_id)
    -> SemIR::TypeInstId {
  auto canon_facet_type_inst_id =
      context.constant_values().GetConstantInstId(facet_type_inst_id);
  if (canon_facet_type_inst_id == SemIR::ErrorInst::TypeInstId) {
    return SemIR::ErrorInst::TypeInstId;
  }

  auto period_self_replacement_id = context.constant_values().Get(self_inst_id);

  auto orig_facet_type =
      context.insts().GetAs<SemIR::FacetType>(canon_facet_type_inst_id);
  const auto& orig_declared_facet_type = context.declared_facet_types().Get(
      orig_facet_type.declared_facet_type_id);

  auto replace_interface = [&](SemIR::SpecificInterface si) {
    return SubstPeriodSelf(context, loc_id, si, period_self_replacement_id);
  };
  auto replace_constraint = [&](SemIR::SpecificNamedConstraint sc) {
    return SubstPeriodSelf(context, loc_id, sc, period_self_replacement_id);
  };
  auto replace_type_impls_interface =
      [&](SemIR::DeclaredFacetType::TypeImplsInterface impls)
      -> SemIR::DeclaredFacetType::TypeImplsInterface {
    auto self = SubstPeriodSelf(context, loc_id,
                                context.constant_values().Get(impls.self_type),
                                period_self_replacement_id);
    auto interface = SubstPeriodSelf(context, loc_id, impls.specific_interface,
                                     period_self_replacement_id);
    return {context.constant_values().GetInstId(self), interface};
  };
  auto replace_type_impls_constraint =
      [&](SemIR::DeclaredFacetType::TypeImplsNamedConstraint impls)
      -> SemIR::DeclaredFacetType::TypeImplsNamedConstraint {
    auto self = SubstPeriodSelf(context, loc_id,
                                context.constant_values().Get(impls.self_type),
                                period_self_replacement_id);
    auto constraint =
        SubstPeriodSelf(context, loc_id, impls.specific_named_constraint,
                        period_self_replacement_id);
    return {context.constant_values().GetInstId(self), constraint};
  };
  auto replace_rewrite = [&](SemIR::DeclaredFacetType::RewriteConstraint r)
      -> SemIR::DeclaredFacetType::RewriteConstraint {
    // The LHS access instruction is not substituted so it keeps its `.Self`.
    // This avoids evaluation replacing it with a concrete value from a final
    // impl, as that would drop the association with the associated constant
    // being rewritten.
    auto rhs = SubstPeriodSelf(context, loc_id,
                               context.constant_values().Get(r.rhs_id),
                               period_self_replacement_id);
    return {r.lhs_id, context.constant_values().GetInstId(rhs)};
  };

  SemIR::DeclaredFacetType declared_facet_type;
  llvm::append_range(
      declared_facet_type.extend_constraints,
      llvm::map_range(orig_declared_facet_type.extend_constraints,
                      replace_interface));
  llvm::append_range(
      declared_facet_type.extend_named_constraints,
      llvm::map_range(orig_declared_facet_type.extend_named_constraints,
                      replace_constraint));
  llvm::append_range(
      declared_facet_type.self_impls_constraints,
      llvm::map_range(orig_declared_facet_type.self_impls_constraints,
                      replace_interface));
  llvm::append_range(
      declared_facet_type.self_impls_named_constraints,
      llvm::map_range(orig_declared_facet_type.self_impls_named_constraints,
                      replace_constraint));
  llvm::append_range(
      declared_facet_type.type_impls_interfaces,
      llvm::map_range(orig_declared_facet_type.type_impls_interfaces,
                      replace_type_impls_interface));
  llvm::append_range(
      declared_facet_type.type_impls_named_constraints,
      llvm::map_range(orig_declared_facet_type.type_impls_named_constraints,
                      replace_type_impls_constraint));
  llvm::append_range(
      declared_facet_type.rewrite_constraints,
      llvm::map_range(orig_declared_facet_type.rewrite_constraints,
                      replace_rewrite));

  declared_facet_type.Canonicalize();
  if (declared_facet_type == orig_declared_facet_type) {
    // Nothing was substituted, keep the original instruction.
    //
    // It is noteworthy that we keep the non-canonical instruction here, since
    // it may have a symbolic value (which is attached to a generic, and can be
    // updated by specifics). Returning the canonical facet type instruction
    // would lose the attachment to the generic which would be incorrect.
    return facet_type_inst_id;
  }

  return AddTypeInst<SemIR::FacetType>(
      context, loc_id,
      {.type_id = SemIR::TypeType::TypeId,
       .declared_facet_type_id =
           context.declared_facet_types().Add(declared_facet_type)});
}

auto IsPeriodSelf(Context& context, SemIR::InstId inst_id, bool canonicalize)
    -> bool {
  return TryGetAsPeriodSelf(context, inst_id, canonicalize).has_value();
}

class FreezeAndThawCallbacks : public SubstInstCallbacks {
 public:
  explicit FreezeAndThawCallbacks(Context* context, bool match_frozen)
      : SubstInstCallbacks(context), match_frozen_(match_frozen) {}
  auto Subst(SemIR::InstId& inst_id) -> SubstResult override {
    if (inst_id == SemIR::TypeType::TypeInstId ||
        inst_id == SemIR::ErrorInst::InstId) {
      return FullySubstituted;
    }

    if (auto found = cache_.Lookup(inst_id)) {
      inst_id = found.value();
      return FullySubstituted;
    }

    // No need to canonicalize, Subst will recurse to find it and we want to
    // preserve structure.
    if (auto period_self =
            TryGetAsPeriodSelf(context(), inst_id, /*canonicalize=*/false)) {
      auto entity_name =
          context().entity_names().Get(period_self->bind.entity_name_id);
      if (!!entity_name.is_frozen_period_self == match_frozen_) {
        entity_name.is_frozen_period_self = !entity_name.is_frozen_period_self;
        auto bind = period_self->bind;
        bind.entity_name_id =
            context().entity_names().AddCanonical(entity_name);
        auto subst_id = Rebuild(inst_id, bind);
        cache_.Insert(inst_id, subst_id);
        inst_id = subst_id;
        // The type of `.Self` may contain another `.Self` as in `Z(.Self) where
        // .Self ...` so we would need to SubstOperands still to get to them.
        // But we just leave them as frozen. When identifying a facet type and
        // substituting in, we will replace the `.Self` value here, which means
        // its frozen type is never used.
        return FullySubstituted;
      }
    }

    return SubstOperands;
  }

  auto ReuseUnchanged(SemIR::InstId orig_inst_id) -> SemIR::InstId override {
    cache_.Insert(orig_inst_id, orig_inst_id);
    return orig_inst_id;
  }

  auto Rebuild(SemIR::InstId orig_inst_id, SemIR::Inst new_inst)
      -> SemIR::InstId override {
    auto inserted = cache_.Insert(orig_inst_id, [&] {
      if (context().constant_values().GetConstantInstId(orig_inst_id) ==
          orig_inst_id) {
        return RebuildNewInst(SemIR::LocId(orig_inst_id), new_inst);
      } else {
        return AddInst(context(), SemIR::LocIdAndInst::RuntimeVerified(
                                      context().sem_ir(),
                                      SemIR::LocId(orig_inst_id), new_inst));
      }
    });
    return inserted.value();
  }

 private:
  // If true, we are finding frozen `.Self` and thawing them. If false, then the
  // inverse.
  bool match_frozen_;

  // Track replacements that have been done, so that we avoid re-evaluating
  // the same instructions repeatedly. Without this, a facet type can create a
  // quadratic number of evaluations, as each one produces many more,
  // repeatedly.
  Map<SemIR::InstId, SemIR::InstId, 16> cache_;
};

auto ThawPeriodSelf(Context& context, SemIR::InstId inst_id) -> SemIR::InstId {
  FreezeAndThawCallbacks callbacks(&context, /*match_frozen=*/true);
  return SubstInst(context, inst_id, callbacks);
}

auto FreezePeriodSelf(Context& context, SemIR::ConstantId const_id)
    -> SemIR::ConstantId {
  FreezeAndThawCallbacks callbacks(&context, /*match_frozen=*/false);
  auto inst_id = SubstInst(
      context, context.constant_values().GetInstId(const_id), callbacks);
  return context.constant_values().Get(inst_id);
}

}  // namespace Carbon::Check
