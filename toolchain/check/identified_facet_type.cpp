// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/identified_facet_type.h"

#include "toolchain/base/canonical_value_store_impl.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/base/value_store_impl.h"
#include "toolchain/check/context.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/subst.h"
#include "toolchain/sem_ir/specific_interface.h"

namespace Carbon::Check {

template <typename T>
using LessThanFn = llvm::function_ref<auto(const T&, const T&)->bool>;

template <typename VecT>
static auto SortAndDeduplicate(VecT& vec,
                               LessThanFn<typename VecT::value_type> compare)
    -> void {
  llvm::sort(vec, compare);
  vec.erase(llvm::unique(vec), vec.end());
}

// Canonically ordered by the numerical ids. Matches the order of RequiredLess
// when the `self_facet_value` is the same for all interfaces.
static auto InterfaceLess(const SemIR::SpecificInterface& lhs,
                          const SemIR::SpecificInterface& rhs) -> bool {
  return std::tie(lhs.interface_id.index, lhs.specific_id.index) <
         std::tie(rhs.interface_id.index, rhs.specific_id.index);
}

// Canonically ordered by the numerical ids.
static auto RequiredLess(const IdentifiedFacetType::RequiredImpl& lhs,
                         const IdentifiedFacetType::RequiredImpl& rhs) -> bool {
  return std::tuple(lhs.self_facet.RawValue().index,
                    lhs.interface.RawValue().interface_id.index,
                    lhs.interface.RawValue().specific_id.index) <
         std::tuple(rhs.self_facet.RawValue().index,
                    rhs.interface.RawValue().interface_id.index,
                    rhs.interface.RawValue().specific_id.index);
}

// Canonically ordered by the numerical ids.
static auto RewriteLess(const IdentifiedFacetType::Rewrite& lhs,
                        const IdentifiedFacetType::Rewrite& rhs) -> bool {
  return std::tuple(lhs.lhs.RawValue().index, lhs.rhs.RawValue().index) <
         std::tuple(rhs.lhs.RawValue().index, rhs.rhs.RawValue().index);
}

// Canonically ordered by the numerical ids.
static auto EquivalentLess(const IdentifiedFacetType::Equivalent& lhs,
                           const IdentifiedFacetType::Equivalent& rhs) -> bool {
  return std::tuple(lhs.lhs.RawValue().index, lhs.rhs.RawValue().index) <
         std::tuple(rhs.lhs.RawValue().index, rhs.rhs.RawValue().index);
}

IdentifiedFacetType::IdentifiedFacetType(
    IdentifiedFacetTypeKey key, bool partially_identified,
    llvm::ArrayRef<RequiredImpl> extends,
    llvm::ArrayRef<RequiredImpl> self_impls, llvm::ArrayRef<Rewrite> rewrites,
    llvm::ArrayRef<Equivalent> equivalents)
    : key_(key) {
  required_impls_.reserve(extends.size() + self_impls.size());
  llvm::append_range(required_impls_, extends);
  llvm::append_range(required_impls_, self_impls);
  SortAndDeduplicate(required_impls_, RequiredLess);

  if (partially_identified) {
    // This marks the IdentifiedFacetType as being partially identified, and
    // prevents the key from colliding with a fully identified facet type, or
    // with other partially (but differently) identified facet types, with the
    // same constituents but a more complete set of required interfaces.
    key_.num_require_impls = required_impls_.size();
  }

  // We can drop the self facet for these as an extending impls requirement
  // always contains the identified self.
  llvm::append_range(
      extend_impls_,
      llvm::map_range(extends, [](RequiredImpl req) { return req.interface; }));

  rewrites_.assign(rewrites);
  SortAndDeduplicate(rewrites_, RewriteLess);
  equivalents_.assign(equivalents);
  SortAndDeduplicate(equivalents_, EquivalentLess);
}

class EvaluateSubstCallbacks : public SubstInstCallbacks {
 public:
  explicit EvaluateSubstCallbacks(
      Context* context, SemIR::LocId loc_id,
      SubstIdentifiedWitnessesCallbacks* subst_callbacks = nullptr)
      : SubstInstCallbacks(context),
        loc_id_(loc_id),
        subst_callbacks_(subst_callbacks) {}

  auto Subst(SemIR::InstId& inst_id) -> SubstResult override {
    // SubstPeriodSelf recurses into facet types, so we need to do the same to
    // find those witnesses first. So we can't early out if the inst has a
    // concrete constant value.
    if (inst_id == SemIR::TypeType::TypeInstId ||
        inst_id == SemIR::ErrorInst::InstId) {
      return FullySubstituted;
    }
    return SubstOperands;
  }

  auto ReuseUnchanged(SemIR::InstId orig_inst_id) -> SemIR::InstId override {
    if (auto witness = context().insts().TryGetAs<SemIR::IdentifiedWitness>(
            orig_inst_id)) {
      return SubstIdentifiedWitness(*witness);
    }
    if (auto witness =
            context().insts().TryGetAs<SemIR::ImplSelfWitness>(orig_inst_id)) {
      return SubstImplSelfWitness(*witness);
    }
    return orig_inst_id;
  }

  auto Rebuild(SemIR::InstId /*orig_inst_id*/, SemIR::Inst new_inst)
      -> SemIR::InstId override {
    if (auto witness = new_inst.TryAs<SemIR::IdentifiedWitness>()) {
      return SubstIdentifiedWitness(*witness);
    }
    if (auto witness = new_inst.TryAs<SemIR::ImplSelfWitness>()) {
      return SubstImplSelfWitness(*witness);
    }
    return RebuildNewInst(loc_id_, new_inst);
  }

  auto RebuildSpecific(SemIR::SpecificId orig_specific_id,
                       SemIR::InstBlockId args_id)
      -> SemIR::SpecificId override {
    auto generic_id = context().specifics().Get(orig_specific_id).generic_id;
    // Drop the `identified` flag from any specifics we create here, as we have
    // substituted the IdentifiedWitness instructions away.
    return context().specifics().GetOrAdd(generic_id, args_id,
                                          /*identified=*/false);
  }

 private:
  auto SubstIdentifiedWitness(SemIR::IdentifiedWitness witness)
      -> SemIR::InstId {
    if (subst_callbacks_) {
      auto replaced_id = subst_callbacks_->ReplaceIdentified(witness);
      if (replaced_id.has_value()) {
        CARBON_CHECK(
            !context().insts().Is<SemIR::IdentifiedWitness>(replaced_id));
        return replaced_id;
      }
    }

    auto const_id = EvalOrAddInst<SemIR::LookupImplWitness>(
        context(), loc_id_,
        {.type_id = witness.type_id,
         .query_self_inst_id = witness.query_self_inst_id,
         .query_specific_interface_id = witness.query_specific_interface_id});
    auto inst_id = context().constant_values().GetInstId(const_id);
    CARBON_CHECK(!context().insts().Is<SemIR::IdentifiedWitness>(inst_id));
    return inst_id;
  }

  auto SubstImplSelfWitness(SemIR::ImplSelfWitness witness) -> SemIR::InstId {
    if (subst_callbacks_) {
      auto replaced_id = subst_callbacks_->ReplaceImplSelfWitness(witness);
      if (replaced_id.has_value()) {
        return replaced_id;
      }
    }
    return RebuildNewInst(loc_id_, witness);
  }

  SemIR::LocId loc_id_;
  SubstIdentifiedWitnessesCallbacks* subst_callbacks_;
};

auto IdentifiedFacetType::EvaluateImplWitnessesToGetImplAsTarget(
    Context& context, SemIR::LocId loc_id) const -> ImplAsTarget {
  llvm::SmallVector<SemIR::SpecificInterface> targets;
  targets.reserve(extend_impls_.size());
  for (auto interface : extend_impls_) {
    targets.push_back(interface.EvaluateIdentifiedWitnesses(context, loc_id));
  }
  SortAndDeduplicate(
      targets, [](SemIR::SpecificInterface lhs, SemIR::SpecificInterface rhs) {
        return std::tie(lhs.interface_id.index, lhs.specific_id.index) <
               std::tie(rhs.interface_id.index, rhs.specific_id.index);
      });
  return ImplAsTarget(std::move(targets));
}

auto ConstantWithIdentifiedWitnesses::EvaluateIdentifiedWitnesses(
    Context& context, SemIR::LocId loc_id) const -> SemIR::ConstantId {
  if (context.eval_impl_lookup_to_identified_witness()) {
    return const_id_;
  }
  EvaluateSubstCallbacks callbacks(&context, loc_id);
  auto inst_id = SubstInst(
      context, context.constant_values().GetInstId(const_id_), callbacks);
  return context.constant_values().Get(inst_id);
}

auto ConstantWithIdentifiedWitnesses::SubstIdentifiedWitnesses(
    Context& context, SemIR::LocId loc_id,
    SubstIdentifiedWitnessesCallbacks& callbacks) const -> SemIR::ConstantId {
  // SubstIdentifiedWitnesses allows the caller to cleverly replace witnesses
  // coming from identify with their own. If we do this in the middle of
  // identify, it means there is another caller waiting for identify and they
  // may also want to do their own replacements. This would be problematic! But
  // since during identify we only create specifics, and resolving them through
  // eval should not end up here, we make note of that.
  CARBON_CHECK(!context.eval_impl_lookup_to_identified_witness(),
               "SubstIdentifiedWitnesses called during identify");

  EvaluateSubstCallbacks subst_callbacks(&context, loc_id, &callbacks);
  auto inst_id = SubstInst(
      context, context.constant_values().GetInstId(const_id_), subst_callbacks);
  return context.constant_values().Get(inst_id);
}

auto InterfaceWithIdentifiedWitnesses::EvaluateIdentifiedWitnesses(
    Context& context, SemIR::LocId loc_id) const -> SemIR::SpecificInterface {
  if (context.eval_impl_lookup_to_identified_witness()) {
    return interface_;
  }
  EvaluateSubstCallbacks callbacks(&context, loc_id);
  auto specific_id = interface_.specific_id;
  if (specific_id.has_value()) {
    const auto& specific = context.specifics().Get(specific_id);
    llvm::SmallVector<SemIR::InstId> args(
        context.inst_blocks().Get(specific.args_id));
    for (auto& arg_id : args) {
      arg_id = SubstInst(context, arg_id, callbacks);
    }
    specific_id = MakeSpecific(context, loc_id, specific.generic_id, args);
  }
  return {.interface_id = interface_.interface_id, .specific_id = specific_id};
}

auto InterfaceWithIdentifiedWitnesses::SubstIdentifiedWitnesses(
    Context& context, SemIR::LocId loc_id,
    SubstIdentifiedWitnessesCallbacks& callbacks) const
    -> SemIR::SpecificInterface {
  // SubstIdentifiedWitnesses allows the caller to cleverly replace witnesses
  // coming from identify with their own. If we do this in the middle of
  // identify, it means there is another caller waiting for identify and they
  // may also want to do their own replacements. This would be problematic! But
  // since during identify we only create specifics, and resolving them through
  // eval should not end up here, we make note of that.
  CARBON_CHECK(!context.eval_impl_lookup_to_identified_witness(),
               "SubstIdentifiedWitnesses called during identify");

  EvaluateSubstCallbacks subst_callbacks(&context, loc_id, &callbacks);
  auto specific_id = interface_.specific_id;
  if (specific_id.has_value()) {
    const auto& specific = context.specifics().Get(specific_id);
    llvm::SmallVector<SemIR::InstId> args(
        context.inst_blocks().Get(specific.args_id));
    for (auto& arg_id : args) {
      arg_id = SubstInst(context, arg_id, subst_callbacks);
    }
    specific_id = MakeSpecific(context, loc_id, specific.generic_id, args);
  }
  return {.interface_id = interface_.interface_id, .specific_id = specific_id};
}

auto EvaluateIdentifiedWitnesses(Context& context, SemIR::LocId loc_id,
                                 SemIR::SpecificInterface interface)
    -> SemIR::SpecificInterface {
  return InterfaceWithIdentifiedWitnesses(interface)
      .EvaluateIdentifiedWitnesses(context, loc_id);
}

auto AddCanonicalWitnessesBlock(Context& context,
                                llvm::SmallVector<SemIR::InstId>& witnesses)
    -> SemIR::InstBlockId {
  // Small blocks don't need to be sorted.
  if (witnesses.size() <= 1) {
    return context.inst_blocks().AddCanonical(witnesses);
  }

  llvm::SmallVector<std::pair<SemIR::SpecificInterface, SemIR::InstId>>
      sortable;
  sortable.reserve(witnesses.size());

  // Produce the sorted order based on the witness's SpecificInterface.
  for (auto witness_id : witnesses) {
    auto inst = context.insts().Get(witness_id);
    CARBON_KIND_SWITCH(inst) {
      case CARBON_KIND(SemIR::CustomWitness witness): {
        sortable.push_back({context.specific_interfaces().Get(
                                witness.query_specific_interface_id),
                            witness_id});
        break;
      }
      case CARBON_KIND(SemIR::ImplWitness witness): {
        auto table = context.insts().GetAs<SemIR::ImplWitnessTable>(
            witness.witness_table_id);
        sortable.push_back(
            {context.impls().Get(table.impl_id).interface, witness_id});
        break;
      }
      case CARBON_KIND(SemIR::LookupImplWitness witness): {
        sortable.push_back({context.specific_interfaces().Get(
                                witness.query_specific_interface_id),
                            witness_id});
        break;
      }
      default:
        CARBON_FATAL("Unhandled inst: {0}", inst);
    }
  }
  // This matches the sort order of IdentifiedFacetType::required_interfaces,
  // which is the order of the witnesses returned from impl lookup, and is
  // canonical order in which the witnesses must appear for a given facet type
  // so that ImplWitnessAccess can find the appropriate witness.
  llvm::sort(sortable, [](auto& lhs, auto& rhs) {
    return InterfaceLess(lhs.first, rhs.first);
  });

  // Update the original list with the new order (reusing to avoid an
  // allocation).
  for (auto [witness_id, sortable_entry] :
       llvm::zip_equal(witnesses, sortable)) {
    witness_id = sortable_entry.second;
  }

  return context.inst_blocks().AddCanonical(witnesses);
}

}  // namespace Carbon::Check

namespace Carbon {
template class CanonicalValueStore<
    Check::IdentifiedFacetTypeId, Check::IdentifiedFacetTypeKey,
    Tag<SemIR::CheckIRId>, Check::IdentifiedFacetType>;
template class ValueStore<Check::IdentifiedFacetTypeId,
                          Check::IdentifiedFacetType, Tag<SemIR::CheckIRId>>;
}  // namespace Carbon
