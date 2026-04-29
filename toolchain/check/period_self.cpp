// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/period_self.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/control_flow.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/facet_type.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

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
