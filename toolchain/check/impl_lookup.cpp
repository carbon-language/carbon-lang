// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/impl_lookup.h"

#include <algorithm>
#include <functional>
#include <utility>
#include <variant>

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/cpp/impl_lookup.h"
#include "toolchain/check/custom_witness.h"
#include "toolchain/check/deduce.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/impl.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/period_self.h"
#include "toolchain/check/subst.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/check/type_structure.h"
#include "toolchain/sem_ir/facet_type_info.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/impl.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/type_iterator.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Returns IRs which are allowed to define an `impl` involving the arguments.
// This is limited by the orphan rule.
static auto FindAssociatedImportIRs(
    Context& context, SemIR::ConstantId query_self_const_id,
    SemIR::SpecificInterface query_specific_interface)
    -> llvm::SmallVector<SemIR::ImportIRId> {
  llvm::SmallVector<SemIR::ImportIRId> result;

  // Add an entity to our result.
  auto add_entity = [&](const SemIR::EntityWithParamsBase& entity) {
    // We will look for impls in the import IR associated with the first owning
    // declaration.
    auto decl_id = entity.first_owning_decl_id;
    if (!decl_id.has_value()) {
      return;
    }

    auto import_ir_inst = GetCanonicalImportIRInst(context, decl_id);
    const auto* sem_ir = &context.sem_ir();
    if (import_ir_inst.ir_id().has_value()) {
      sem_ir = context.import_irs().Get(import_ir_inst.ir_id()).sem_ir;
    }

    // For an instruction imported from C++, `GetCanonicalImportIRInst` returns
    // the final Carbon import instruction, so go one extra step to check for a
    // C++ import.
    if (auto import_ir_inst_id =
            sem_ir->insts().GetImportSource(import_ir_inst.inst_id());
        import_ir_inst_id.has_value()) {
      result.push_back(
          sem_ir->import_ir_insts().Get(import_ir_inst_id).ir_id());
    } else if (import_ir_inst.ir_id().has_value()) {
      result.push_back(import_ir_inst.ir_id());
    }
  };

  llvm::SmallVector<SemIR::InstId> worklist;

  // Push the contents of an instruction block onto our worklist.
  auto push_block = [&](SemIR::InstBlockId block_id) {
    if (block_id.has_value()) {
      llvm::append_range(worklist, context.inst_blocks().Get(block_id));
    }
  };

  // Add the arguments of a specific to the worklist.
  auto push_args = [&](SemIR::SpecificId specific_id) {
    if (specific_id.has_value()) {
      push_block(context.specifics().Get(specific_id).args_id);
    }
  };

  worklist.push_back(context.constant_values().GetInstId(query_self_const_id));
  add_entity(context.interfaces().Get(query_specific_interface.interface_id));
  push_args(query_specific_interface.specific_id);

  while (!worklist.empty()) {
    auto inst_id = worklist.pop_back_val();

    // Visit the operands of the constant.
    auto inst = context.insts().Get(inst_id);
    for (auto arg : {inst.arg0_and_kind(), inst.arg1_and_kind()}) {
      CARBON_KIND_SWITCH(arg) {
        case CARBON_KIND(SemIR::InstId inst_id): {
          if (inst_id.has_value()) {
            worklist.push_back(inst_id);
          }
          break;
        }
        case CARBON_KIND(SemIR::TypeInstId inst_id): {
          if (inst_id.has_value()) {
            worklist.push_back(inst_id);
          }
          break;
        }
        case CARBON_KIND(SemIR::InstBlockId inst_block_id): {
          push_block(inst_block_id);
          break;
        }
        case CARBON_KIND(SemIR::ClassId class_id): {
          add_entity(context.classes().Get(class_id));
          break;
        }
        case CARBON_KIND(SemIR::InterfaceId interface_id): {
          add_entity(context.interfaces().Get(interface_id));
          break;
        }
        case CARBON_KIND(SemIR::FacetTypeId facet_type_id): {
          const auto& facet_type_info =
              context.facet_types().Get(facet_type_id);
          for (const auto& impl : facet_type_info.extend_constraints) {
            add_entity(context.interfaces().Get(impl.interface_id));
            push_args(impl.specific_id);
          }
          for (const auto& impl : facet_type_info.self_impls_constraints) {
            add_entity(context.interfaces().Get(impl.interface_id));
            push_args(impl.specific_id);
          }
          break;
        }
        case CARBON_KIND(SemIR::FunctionId function_id): {
          add_entity(context.functions().Get(function_id));
          break;
        }
        case CARBON_KIND(SemIR::SpecificId specific_id): {
          push_args(specific_id);
          break;
        }
        default: {
          break;
        }
      }
    }
  }

  // Deduplicate.
  llvm::sort(result, [](SemIR::ImportIRId a, SemIR::ImportIRId b) {
    return a.index < b.index;
  });
  result.erase(llvm::unique(result), result.end());

  return result;
}

// Returns true if a cycle was found and diagnosed.
static auto FindAndDiagnoseImplLookupCycle(
    Context& context, llvm::SmallVector<Context::ImplLookupStackEntry>& stack,
    SemIR::LocId loc_id, SemIR::ConstantId query_self_const_id,
    SemIR::ConstantId query_facet_type_const_id, bool diagnose) -> bool {
  // Deduction of the interface parameters can do further impl lookups, and we
  // need to ensure we terminate.
  //
  // https://docs.carbon-lang.dev/docs/design/generics/details.html#acyclic-rule
  // - We look for violations of the acyclic rule by seeing if a previous lookup
  //   had all the same type inputs.
  // - The `query_facet_type_const_id` encodes the entire facet type being
  //   looked up, including any specific parameters for a generic interface.
  //
  // TODO: Implement the termination rule, which requires looking at the
  // complexity of the types on the top of (or throughout?) the stack:
  // https://docs.carbon-lang.dev/docs/design/generics/details.html#termination-rule
  for (auto [i, entry] : llvm::enumerate(stack)) {
    if (entry.query_self_const_id == query_self_const_id &&
        entry.query_facet_type_const_id == query_facet_type_const_id) {
      if (diagnose && !stack.back().diagnosed_cycle) {
        auto facet_type_type_id = context.types().GetTypeIdForTypeConstantId(
            query_facet_type_const_id);
        CARBON_DIAGNOSTIC(ImplLookupCycle, Error,
                          "cycle found in search for impl of {0} for type {1}",
                          SemIR::TypeId, SemIR::TypeId);
        auto builder = context.emitter().Build(
            loc_id, ImplLookupCycle, facet_type_type_id,
            context.types().GetTypeIdForTypeConstantId(query_self_const_id));
        for (const auto& active_entry : llvm::drop_begin(stack, i)) {
          if (active_entry.impl_loc.has_value()) {
            CARBON_DIAGNOSTIC(ImplLookupCycleNote, Note,
                              "determining if this impl clause matches", );
            builder.Note(active_entry.impl_loc, ImplLookupCycleNote);
          }
        }
        builder.Emit();
      }

      stack.back().diagnosed_cycle = true;
      return true;
    }
  }
  return false;
}

struct RequiredImplsFromConstraint {
  llvm::ArrayRef<SemIR::IdentifiedFacetType::RequiredImpl> req_impls;
  bool other_requirements;
};

// Gets the set of `SpecificInterface`s that are required by a facet type
// (as a constant value), and any special requirements.
static auto GetRequiredImplsFromConstraint(
    Context& context, SemIR::LocId loc_id,
    SemIR::ConstantId query_self_const_id,
    SemIR::ConstantId query_facet_type_const_id, bool diagnose)
    -> std::optional<RequiredImplsFromConstraint> {
  auto facet_type_inst_id =
      context.constant_values().GetInstId(query_facet_type_const_id);
  auto facet_type_inst =
      context.insts().GetAs<SemIR::FacetType>(facet_type_inst_id);
  const auto& facet_type_info =
      context.facet_types().Get(facet_type_inst.facet_type_id);

  auto identified_id = RequireIdentifiedFacetType(
      context, loc_id, query_self_const_id, facet_type_inst,
      [&](auto& builder) {
        CARBON_DIAGNOSTIC(ImplLookupInUnidentifiedFacetType, Context,
                          "facet type {0} can not be identified", InstIdAsType);
        builder.Context(loc_id, ImplLookupInUnidentifiedFacetType,
                        facet_type_inst_id);
      },
      diagnose);
  if (!identified_id.has_value()) {
    return std::nullopt;
  }
  return {
      {.req_impls =
           context.identified_facet_types().Get(identified_id).required_impls(),
       .other_requirements = facet_type_info.other_requirements}};
}

static auto TreatImplAsFinal(Context& context, const SemIR::Impl& impl)
    -> bool {
  // Lookups for the impl inside its own definition treat the impl as final.
  // Nothing can specialize those lookups further, and it resolves any accesses
  // of associated constants to their concrete values.
  return IsImplEffectivelyFinal(context, impl) || impl.is_being_defined();
}

// Given a (possibly generic) `impl`, deduce a specific `impl` from the query
// self and specific for the interface. Return the witness ID of the `impl` of
// the resulting specific `impl`, if its specific interface matches the query.
//
// Note the witness also has the specific for the `impl` applied to it.
static auto TryGetSpecificWitnessIdForImpl(
    Context& context, SemIR::LocId loc_id,
    SemIR::ConstantId query_self_const_id,
    const SemIR::SpecificInterface& interface, const SemIR::Impl& impl)
    -> SemIR::ConstantId {
  // The impl may have generic arguments, in which case we need to deduce them
  // to find what they are given the specific type and interface query. We use
  // that specific to map values in the impl to the deduced values.
  auto specific_id = SemIR::SpecificId::None;
  if (impl.generic_id.has_value()) {
    specific_id = DeduceImplArguments(
        context, loc_id, impl, query_self_const_id, interface.specific_id);
    if (!specific_id.has_value()) {
      return SemIR::ConstantId::None;
    }
  }

  // The self type of the impl must match the type in the query, or this is an
  // `impl T as ...` for some other type `T` and should not be considered.
  auto noncanonical_deduced_self_const_id = SemIR::GetConstantValueInSpecific(
      context.sem_ir(), specific_id, impl.self_id);

  // In a generic `impl forall` the self type can be a FacetAccessType, which
  // will not be the same constant value as a query facet value. We move through
  // to the facet value here, and if the query was a FacetAccessType we did the
  // same there so they still match.
  auto deduced_self_const_id =
      GetCanonicalFacetOrTypeValue(context, noncanonical_deduced_self_const_id);
  if (query_self_const_id != deduced_self_const_id) {
    return SemIR::ConstantId::None;
  }

  // The impl's constraint is a facet type which it is implementing for the self
  // type: the `I` in `impl ... as I`. The deduction step may be unable to be
  // fully applied to the types in the constraint and result in an error here,
  // in which case it does not match the query.
  auto deduced_constraint_id = SemIR::GetConstantValueInSpecific(
      context.sem_ir(), specific_id, impl.constraint_id);
  if (deduced_constraint_id == SemIR::ErrorInst::ConstantId) {
    return SemIR::ConstantId::None;
  }

  auto deduced_constraint_facet_type_id =
      context.constant_values()
          .GetInstAs<SemIR::FacetType>(deduced_constraint_id)
          .facet_type_id;
  const auto& deduced_constraint_facet_type_info =
      context.facet_types().Get(deduced_constraint_facet_type_id);
  CARBON_CHECK(deduced_constraint_facet_type_info.extend_constraints.size() ==
               1);

  if (deduced_constraint_facet_type_info.other_requirements) {
    return SemIR::ConstantId::None;
  }

  // The specifics in the queried interface must match the deduced specifics in
  // the impl's constraint facet type.
  auto impl_interface_specific_id =
      deduced_constraint_facet_type_info.extend_constraints[0].specific_id;
  auto query_interface_specific_id = interface.specific_id;
  if (impl_interface_specific_id != query_interface_specific_id) {
    return SemIR::ConstantId::None;
  }

  LoadImportRef(context, impl.witness_id);
  if (!impl.is_being_defined() && specific_id.has_value()) {
    // If the impl definition can be resolved, eval will do it immediately;
    // otherwise, it can be resolved by further specialization. This is used to
    // resolve dependency chains when `MakeFinal` is returned without a concrete
    // definition; particularly final impls with symbolic constants.
    //
    // Note we do not do this for lookups _inside_ the definition of the impl,
    // as that creates a cycle where resolving the definition must resolve the
    // definition.
    AddInstInNoBlock(
        context, loc_id,
        SemIR::RequireSpecificDefinition{
            .type_id = GetSingletonType(
                context, SemIR::RequireSpecificDefinitionType::TypeInstId),
            .specific_id = specific_id});
  }

  return SemIR::GetConstantValueInSpecific(context.sem_ir(), specific_id,
                                           impl.witness_id);
}

struct FacetWitnessSource {
  // A facet, of type FacetType. If this is a FacetValue, we may be able to get
  // a concrete witness out of it.
  SemIR::ConstantId facet_const_id;
  // The set of witnesses this facet provides at least symbolically. Encodes
  // the order of the witnesses for looking for a concrete witness in `facet`
  // if it's a FacetValue.
  SemIR::IdentifiedFacetTypeId identified_facet_type_id;
};

// Collect witnesses from facets in the query. The facets' types in the query
// self are allowed to be partially identified to support the use of `Self`
// inside a named constraint, which can appear anywhere in the query self, e.g.
// as `Self` or as `C(Self)`.
static auto CollectFacetWitnessSources(
    Context& context, SemIR::LocId loc_id,
    SemIR::ConstantId query_self_const_id,
    SemIR::ConstantId query_facet_type_const_id)
    -> llvm::SmallVector<FacetWitnessSource> {
  auto query_self_inst_id =
      context.constant_values().GetInstId(query_self_const_id);
  auto query_facet_type_inst_id =
      context.constant_values().GetInstId(query_facet_type_const_id);

  llvm::SmallVector<FacetWitnessSource> witnesses;

  auto push_facet = [&](SemIR::InstId facet, bool allow_partially_identified) {
    auto facet_const_id = context.constant_values().Get(facet);

    auto type_id = context.insts().Get(facet).type_id();
    if (type_id != SemIR::TypeType::TypeId) {
      auto facet_type = context.types().GetAs<SemIR::FacetType>(type_id);
      auto identified_id =
          TryToIdentifyFacetType(context, loc_id, facet_const_id, facet_type,
                                 allow_partially_identified);
      if (identified_id.has_value()) {
        witnesses.push_back({.facet_const_id = facet_const_id,
                             .identified_facet_type_id = identified_id});
      }
    }
  };

  auto collect_facets = [&](SemIR::TypeIterator& iter,
                            bool allow_partially_identified) {
    for (auto done = false; !done;) {
      auto next = iter.Next();
      CARBON_KIND_SWITCH(next.any) {
        using Step = SemIR::TypeIterator::Step;
        case CARBON_KIND(Step::Done _): {
          done = true;
          break;
        }
        case CARBON_KIND(Step::TypeWrapper wrapper): {
          switch (wrapper.kind) {
            case Step::TypeWrapper::FacetValue:
              // We want to store FacetValues since they come with final
              // witnesses, regardless of whether they internally hold a
              // concrete type or a symbolic one (with non-final witnesses of
              // its own).
              push_facet(wrapper.inst_id, allow_partially_identified);
              break;
            case Step::TypeWrapper::ImplWitnessAccess:
              // We want to store ImplWitnessAccess because the associated
              // constant may be a facet with witnesses.
              push_facet(wrapper.inst_id, allow_partially_identified);
              break;
          }
          break;
        }
        case CARBON_KIND(Step::SymbolicType symbolic): {
          // We want to store SymbolicTypes. They may occur inside or outside of
          // a FacetValue. If inside a FacetValue, they may provide a non-final
          // witness for more things than the FacetValue would, since the
          // FacetValue represents a conversion which can lose part of the facet
          // type.
          push_facet(symbolic.facet, allow_partially_identified);
          break;
        }
        default:
          break;
      }
    }
  };

  {
    SemIR::TypeIterator iter(&context.sem_ir());
    iter.Add(query_self_inst_id);
    collect_facets(iter, /*allow_partially_identified=*/true);
  }

  {
    SemIR::TypeIterator iter(&context.sem_ir());
    iter.Add(context.insts().GetAs<SemIR::FacetType>(query_facet_type_inst_id));
    collect_facets(iter, /*allow_partially_identified=*/false);
  }

  if (!context.where_stack().empty()) {
    // Grab witnesses from any `impls` constraints in the current `where`
    // expression.
    //
    // The `where` expression may be nested inside another `where`, but the
    // inner `where` expression is checked before the outer, so it needs to be
    // self-consistent and provide any `impls` constraints required by other
    // constraints.
    const auto& impls = context.where_stack().back().impls;
    for (auto [self_const_id, facet_type_const_id] : impls) {
      auto canon_self_const_id =
          GetCanonicalFacetOrTypeValue(context, self_const_id);
      // TypeType is never stored in the impls stack, so we always have a
      // FacetType.
      auto facet_type = context.constant_values().GetInstAs<SemIR::FacetType>(
          facet_type_const_id);
      auto identified_id = TryToIdentifyFacetType(
          context, loc_id, canon_self_const_id, facet_type,
          /*allow_partially_identified=*/true);
      if (identified_id.has_value()) {
        witnesses.push_back({.facet_const_id = canon_self_const_id,
                             .identified_facet_type_id = identified_id});
      }
    }
  }

  return witnesses;
}

// Given a query `orig_inst_self` and `orig_interface`, try find a matching
// witness from impl lookup to use for the query.
static auto TryFindMatchingWitnessFromImplLookup(
    Context& context, SemIR::LocId loc_id,
    SemIR::ConstantId canonical_query_self_const_id,
    llvm::ArrayRef<SemIR::IdentifiedFacetType::RequiredImpl> req_impls,
    llvm::ArrayRef<SemIR::InstId> found_witness_inst_ids,
    SemIR::ConstantId orig_const_self, SemIR::SpecificInterface orig_interface)
    -> SemIR::InstId {
  // The `req_impls` come from an IdentifiedFacetType so they have `.Self`
  // replaced. We need to do the same for the self and interface in the
  // `orig_witness` for comparing with them.
  orig_const_self = SubstPeriodSelf(context, loc_id, orig_const_self,
                                    canonical_query_self_const_id);
  orig_interface = SubstPeriodSelf(context, loc_id, orig_interface,
                                   canonical_query_self_const_id);

  // Witnesses have a canonicalized self value. Perform the same
  // canonicalization here so that we can compare them.
  orig_const_self =
      GetCanonicalQuerySelfForLookupImplWitness(context, orig_const_self);

  for (auto [req_impl, found_witness_inst_id] :
       llvm::zip_equal(req_impls, found_witness_inst_ids)) {
    auto [req_const_self, req_interface] = req_impl;
    if (req_const_self == orig_const_self && req_interface == orig_interface) {
      return found_witness_inst_id;
    }
  }
  return SemIR::InstId::None;
}

static auto VerifyQueryFacetTypeConstraints(
    Context& context, SemIR::LocId loc_id,
    SemIR::ConstantId query_self_const_id,
    SemIR::ConstantId query_facet_type_const_id,
    llvm::ArrayRef<SemIR::IdentifiedFacetType::RequiredImpl> req_impls,
    llvm::ArrayRef<SemIR::InstId> witness_inst_ids) -> bool {
  const auto& facet_type_info = context.facet_types().Get(
      context.constant_values()
          .GetInstAs<SemIR::FacetType>(query_facet_type_const_id)
          .facet_type_id);

  if (!facet_type_info.rewrite_constraints.empty()) {
    auto rebuild = [&](SemIR::Inst new_inst) -> SemIR::InstId {
      // When rebuilding a witness where `.Self` was replaced, use a witness we
      // found in impl lookup instead of performing impl lookup again.
      if (auto lookup = new_inst.TryAs<SemIR::LookupImplWitness>()) {
        auto witness = TryFindMatchingWitnessFromImplLookup(
            context, loc_id, query_self_const_id, req_impls, witness_inst_ids,
            context.constant_values().Get(lookup->query_self_inst_id),
            context.specific_interfaces().Get(
                lookup->query_specific_interface_id));
        if (witness.has_value()) {
          return witness;
        }
      }

      return SemIR::InstId::None;
    };

    for (const auto& rewrite : facet_type_info.rewrite_constraints) {
      // Replace `.Self` in rewrite constraints with the query self in order to
      // find the provided values of rewrite constraints from the query. This
      // includes replacing `.Self` in LookupImplWitness instructions.
      //
      // When we have found a witness in impl lookup for the query in a
      // LookupImplWitness insts, we need to use that witness directly instead
      // of rebuilding (and reevaluating) the LookupImplWitness which will
      // execute another impl lookup.

      auto lhs_id = context.constant_values().GetInstId(SubstPeriodSelf(
          context, loc_id, context.constant_values().Get(rewrite.lhs_id),
          query_self_const_id, SubstPeriodSelfBehaviour::All, rebuild));
      auto rhs_id = context.constant_values().GetInstId(SubstPeriodSelf(
          context, loc_id, context.constant_values().Get(rewrite.rhs_id),
          query_self_const_id, SubstPeriodSelfBehaviour::All, rebuild));

      if (lhs_id != rhs_id) {
        // TODO: Provide a diagnostic note and location for which rewrite
        // constraint was not satisfied, if a diagnostic is going to be
        // displayed for the LookupImplWitness() failure. This may require
        // plumbing through a callback that lets us add a Note to another
        // diagnostic.
        return false;
      }
    }
  }

  // TODO: Validate that the witnesses satisfy the other requirements in the
  // `facet_type_info`.

  return true;
}

// Returns whether the query is concrete, it is false if the self type or
// interface specifics have a symbolic dependency.
static auto QueryIsConcrete(Context& context, SemIR::ConstantId self_const_id,
                            const SemIR::SpecificInterface& specific_interface)
    -> bool {
  if (!self_const_id.is_concrete()) {
    return false;
  }
  if (!specific_interface.specific_id.has_value()) {
    return true;
  }
  auto args_id =
      context.specifics().Get(specific_interface.specific_id).args_id;
  for (auto inst_id : context.inst_blocks().Get(args_id)) {
    if (!context.constant_values().Get(inst_id).is_concrete()) {
      return false;
    }
  }
  return true;
}

namespace {
// A class to filter imported impls based on whether they could possibly match a
// query, prior to importing them. For now we only consider impls that are for
// an interface that's being queried.
//
// TODO: There's a lot more we could do to filter out impls that can't possibly
// match.
class ImportImplFilter {
 public:
  explicit ImportImplFilter(Context& context, SemIR::ImportIRId import_ir_id,
                            SemIR::SpecificInterface interface)
      : context_(&context),
        interface_id_(interface.interface_id),
        import_ir_id_(import_ir_id),
        import_ir_(context_->import_irs().Get(import_ir_id).sem_ir),
        cached_import_interface_id_(SemIR::InterfaceId::None) {}

  // Returns whether the given impl is potentially relevant for the current
  // query.
  auto IsRelevantImpl(SemIR::ImplId import_impl_id) -> bool {
    auto impl_interface_id =
        import_ir_->impls().Get(import_impl_id).interface.interface_id;
    if (!impl_interface_id.has_value()) {
      // This indicates that an error occurred when type-checking the impl.
      // TODO: Use an explicit error value for this rather than None.
      return false;
    }
    return IsRelevantInterface(impl_interface_id);
  }

 private:
  // Returns whether an impl for the given interface might be relevant to the
  // current query.
  auto IsRelevantInterface(SemIR::InterfaceId import_interface_id) -> bool {
    if (!cached_import_interface_id_.has_value()) {
      if (IsSameInterface(import_interface_id, interface_id_)) {
        cached_import_interface_id_ = import_interface_id;
        return true;
      }
    } else if (cached_import_interface_id_ == import_interface_id) {
      return true;
    }
    return false;
  }

  // Returns whether the given interfaces from two different IRs are the same.
  auto IsSameInterface(SemIR::InterfaceId import_interface_id,
                       SemIR::InterfaceId local_interface_id) -> bool {
    // The names must be the same.
    if (import_ir_->names().GetAsStringIfIdentifier(
            import_ir_->interfaces().Get(import_interface_id).name_id) !=
        context_->names().GetAsStringIfIdentifier(
            context_->interfaces().Get(local_interface_id).name_id)) {
      return false;
    }
    // Compare the interfaces themselves.
    // TODO: Should we check the scope of the interface before doing this?
    auto local_version_of_import_interface_id =
        ImportInterface(*context_, import_ir_id_, import_interface_id);
    return local_version_of_import_interface_id == local_interface_id;
  }

  Context* context_;
  // The interface being looked up.
  SemIR::InterfaceId interface_id_;
  // The IR that we are currently importing impls from.
  SemIR::ImportIRId import_ir_id_;
  const SemIR::File* import_ir_;
  // The interface ID of `interface_id_` in `import_ir_`, if known.
  SemIR::InterfaceId cached_import_interface_id_;
};
}  // namespace

struct CandidateImpl {
  const SemIR::Impl* impl;

  // Used for sorting the candidates to find the most-specialized match.
  TypeStructure type_structure;
};

struct CandidateImpls {
  llvm::SmallVector<CandidateImpl> impls;
  bool consider_cpp_candidates = false;
};

// Returns the list of candidates impls for lookup to select from.
static auto CollectCandidateImplsForQuery(
    Context& context, bool final_only, SemIR::ConstantId query_self_const_id,
    const TypeStructure& query_type_structure,
    SemIR::SpecificInterface& query_specific_interface) -> CandidateImpls {
  CandidateImpls candidates;

  auto import_irs = FindAssociatedImportIRs(context, query_self_const_id,
                                            query_specific_interface);

  for (auto import_ir_id : import_irs) {
    // If `Cpp` is an associated package, then we'll instead look for C++
    // operator overloads for certain well-known interfaces.
    if (import_ir_id == SemIR::ImportIRId::Cpp) {
      candidates.consider_cpp_candidates = true;
      continue;
    }

    // Instead of importing all impls, only import ones that are in some way
    // connected to this query.
    ImportImplFilter filter(context, import_ir_id, query_specific_interface);
    for (auto [import_impl_id, _] :
         context.import_irs().Get(import_ir_id).sem_ir->impls().enumerate()) {
      if (filter.IsRelevantImpl(import_impl_id)) {
        // TODO: Track the relevant impls and only consider those ones and any
        // local impls, rather than looping over all impls below.
        ImportImpl(context, import_ir_id, import_impl_id);
      }
    }
  }

  for (auto [id, impl] : context.impls().enumerate()) {
    CARBON_CHECK(impl.witness_id.has_value());

    if (final_only && !TreatImplAsFinal(context, impl)) {
      continue;
    }

    // If the impl's interface_id differs from the query, then this impl can
    // not possibly provide the queried interface.
    if (impl.interface.interface_id != query_specific_interface.interface_id) {
      continue;
    }

    // When the impl's interface_id matches, but the interface is generic, the
    // impl may or may not match based on restrictions in the generic
    // parameters of the impl.
    //
    // As a shortcut, if the impl's constraint is not symbolic (does not
    // depend on any generic parameters), then we can determine whether we match
    // by looking if the specific ids match exactly.
    auto impl_interface_const_id =
        context.constant_values().Get(impl.constraint_id);
    if (!impl_interface_const_id.is_symbolic() &&
        impl.interface.specific_id != query_specific_interface.specific_id) {
      continue;
    }

    // Build the type structure used for choosing the best the candidate.
    auto type_structure =
        BuildTypeStructure(context, impl.self_id, impl.interface);
    if (!type_structure) {
      continue;
    }
    // TODO: We can skip the comparison here if the `impl_interface_const_id` is
    // not symbolic, since when the interface and specific ids match, and they
    // aren't symbolic, the structure will be identical.
    if (!query_type_structure.CompareStructure(
            TypeStructure::CompareTest::IsEqualToOrMoreSpecificThan,
            *type_structure)) {
      continue;
    }

    candidates.impls.push_back({&impl, std::move(*type_structure)});
  }

  auto compare = [](auto& lhs, auto& rhs) -> bool {
    return lhs.type_structure < rhs.type_structure;
  };
  // Stable sort is used so that impls that are seen first are preferred when
  // they have an equal priority ordering.
  // TODO: Allow Carbon code to provide a priority ordering explicitly. For
  // now they have all the same priority, so the priority is the order in
  // which they are found in code.
  llvm::stable_sort(candidates.impls, compare);

  return candidates;
}

class IndexInFacetValue {
 public:
  static const IndexInFacetValue None;
  static const IndexInFacetValue Unstable;

  explicit constexpr IndexInFacetValue(int32_t index) : index_(index) {}

  // Returns whether the value represents a successful attempt to find the index
  // of an interface in a FacetValue. Returns true regardless of whether the
  // index is stable and able to be used or not.
  auto WasFound() const -> bool { return index_ != None.index_; }

  // Gets the stable index which can be used to index into the witness table in
  // a FacetValue, if there is one. Otherwise, returns -1.
  auto GetStableIndex() const -> int32_t {
    if (index_ == Unstable.index_) {
      return None.index_;
    }
    return index_;
  }

 private:
  int32_t index_;
};

inline constexpr auto IndexInFacetValue::None = IndexInFacetValue(-1);
inline constexpr auto IndexInFacetValue::Unstable = IndexInFacetValue(-2);

// Looks in the identified facet type and returns the index of a witness that
// `query_self_const_id` impls `query_specific_interface` in the defined
// interface order for that facet type.
//
// The IdentifiedFacetType must not be partially identified in order to find an
// index, as that implies the interface order is not yet stable. In that case,
// no index will be found.
//
// If the `query_specific_interface` is not part of the facet type, returns -1
// to indicate it was not found.
static auto IndexOfImplWitnessInIdentifiedFacetType(
    Context& context, SemIR::IdentifiedFacetTypeId identified_facet_type_id,
    SemIR::ConstantId query_self_const_id,
    SemIR::SpecificInterface query_specific_interface) -> IndexInFacetValue {
  // The self in the identified facet type is a canonicalized facet value, so we
  // canonicalize the query for comparison.
  auto canonical_query_self_const_id =
      GetCanonicalFacetOrTypeValue(context, query_self_const_id);

  const auto& identified =
      context.identified_facet_types().Get(identified_facet_type_id);
  auto facet_type_req_impls = llvm::enumerate(identified.required_impls());
  auto it = llvm::find_if(facet_type_req_impls, [&](auto e) {
    auto [req_self, req_specific_interface] = e.value();
    return req_self == canonical_query_self_const_id &&
           req_specific_interface == query_specific_interface;
  });
  if (it == facet_type_req_impls.end()) {
    return IndexInFacetValue::None;
  }

  if (identified.partially_identified()) {
    return IndexInFacetValue::Unstable;
  }
  return IndexInFacetValue(static_cast<int32_t>((*it).index()));
}

static auto FindFinalWitnessFromFacetValue(
    Context& context, llvm::ArrayRef<FacetWitnessSource> witness_sources,
    SemIR::ConstantId query_self_const_id,
    SemIR::SpecificInterface query_specific_interface) -> SemIR::InstId {
  for (auto [facet, identified_id] : witness_sources) {
    auto facet_value =
        context.constant_values().TryGetInstAs<SemIR::FacetValue>(facet);
    if (!facet_value) {
      // This facet can not provide a final witness, keep looking.
      continue;
    }

    auto index_in_facet_value = IndexOfImplWitnessInIdentifiedFacetType(
        context, identified_id, query_self_const_id, query_specific_interface);
    auto stable_index = index_in_facet_value.GetStableIndex();
    if (stable_index < 0) {
      // No witness in this facet, keep looking.
      continue;
    }

    auto witness_id = context.inst_blocks().Get(
        facet_value->witnesses_block_id)[stable_index];
    if (!context.insts().Is<SemIR::LookupImplWitness>(witness_id)) {
      // Found a final witness.
      return witness_id;
    }
  }
  return SemIR::InstId::None;
}

static auto FindNonFinalWitness(
    Context& context, SemIR::LocId loc_id,
    llvm::ArrayRef<FacetWitnessSource> witness_sources,
    SemIR::ConstantId query_self_const_id,
    SemIR::SpecificInterface query_specific_interface) -> bool {
  for (auto [_, identified_id] : witness_sources) {
    auto index = IndexOfImplWitnessInIdentifiedFacetType(
        context, identified_id, query_self_const_id, query_specific_interface);
    if (index.WasFound()) {
      return true;
    }
  }

  // TODO: Remove SpecificInterfaceId from LookupCustomWitness apis, switch to
  // just SpecificInterface.
  auto query_specific_interface_id =
      context.specific_interfaces().Add(query_specific_interface);

  // Consider a custom witness for core interfaces.
  // TODO: This needs to expand to more interfaces, and we might want to have
  // that dispatch in custom_witness.cpp instead of here.
  auto core_interface =
      GetCoreInterface(context, query_specific_interface.interface_id);
  if (auto witness_id = LookupCustomWitness(
          context, loc_id, core_interface, query_self_const_id,
          query_specific_interface_id, false)) {
    // If there's a final witness, we would have already found it via evaluating
    // the LookupImplWitness instruction.
    CARBON_CHECK(!witness_id->has_value());
    return true;
  }

  auto query_type_structure = BuildTypeStructure(
      context, context.constant_values().GetInstId(query_self_const_id),
      query_specific_interface);
  // We looked for errors in the query self and facet type already, and we're
  // not dealing with monomorphizations here.
  CARBON_CHECK(query_type_structure, "error in impl lookup query");

  auto candidates = CollectCandidateImplsForQuery(
      context, /*final_only=*/false, query_self_const_id, *query_type_structure,
      query_specific_interface);

  for (const auto& candidate : candidates.impls) {
    const auto& impl = *candidate.impl;
    context.impl_lookup_stack().back().impl_loc = impl.definition_id;

    auto witness_id = TryGetSpecificWitnessIdForImpl(
        context, loc_id, query_self_const_id, query_specific_interface, impl);
    if (witness_id.has_value()) {
      // We looked for errors in the query self and facet type already, and
      // we're not dealing with monomorphizations here.
      CARBON_CHECK(witness_id != SemIR::ErrorInst::ConstantId,
                   "error in impl lookup query");
      return true;
    }
  }

  // C++ interop only provides final witnesses, so we don't look for a witness
  // from C++ here. Those are found in eval of the `LookupImplWitness`
  // instruction.

  return false;
}

auto LookupImplWitness(Context& context, SemIR::LocId loc_id,
                       SemIR::ConstantId query_self_const_id,
                       SemIR::ConstantId query_facet_type_const_id,
                       bool diagnose) -> SemIR::InstBlockIdOrError {
  if (query_self_const_id == SemIR::ErrorInst::ConstantId ||
      query_facet_type_const_id == SemIR::ErrorInst::ConstantId) {
    return SemIR::InstBlockIdOrError::MakeError();
  }

  {
    // The query self value is a type value or a facet value.
    auto query_self_type_id =
        context.insts()
            .Get(context.constant_values().GetInstId(query_self_const_id))
            .type_id();
    CARBON_CHECK((context.types().IsOneOf<SemIR::TypeType, SemIR::FacetType>(
        query_self_type_id)));
    // The query facet type value is indeed a facet type.
    CARBON_CHECK(context.constant_values().InstIs<SemIR::FacetType>(
        query_facet_type_const_id));
  }

  auto req_impls_from_constraint =
      GetRequiredImplsFromConstraint(context, loc_id, query_self_const_id,
                                     query_facet_type_const_id, diagnose);
  if (!req_impls_from_constraint) {
    return SemIR::InstBlockIdOrError::MakeError();
  }
  auto [req_impls, other_requirements] = *req_impls_from_constraint;
  if (other_requirements) {
    // TODO: Remove this when other requirements go away.
    return SemIR::InstBlockId::None;
  }
  if (req_impls.empty()) {
    return SemIR::InstBlockId::Empty;
  }

  // Find all the facets in the query and their witnesses.
  auto witness_sources = CollectFacetWitnessSources(
      context, loc_id, query_self_const_id, query_facet_type_const_id);

  // Cycles are diagnosed even if they are found when diagnostics are otherwise
  // being suppressed (such as during deduce).
  if (FindAndDiagnoseImplLookupCycle(context, context.impl_lookup_stack(),
                                     loc_id, query_self_const_id,
                                     query_facet_type_const_id, true)) {
    return SemIR::InstBlockIdOrError::MakeError();
  }

  auto& stack = context.impl_lookup_stack();
  stack.push_back({
      .query_self_const_id = query_self_const_id,
      .query_facet_type_const_id = query_facet_type_const_id,
      .diagnosed_cycle = stack.empty() ? false : stack.back().diagnosed_cycle,
  });
  // We need to find a witness for each self+interface pair in `req_impls`.
  //
  // Every consumer of a facet type needs to agree on the order of interfaces
  // used for its witnesses, which is done by following the order in the
  // IdentifiedFacetType of the query facet type, and this is represented in the
  // order of the interfaces in `req_impls`.
  llvm::SmallVector<SemIR::InstId> result_witness_ids;
  for (const auto& req_impl : req_impls) {
    // If the self facet contains a final witness for the required interface, we
    // use that and avoid any further work. This is strictly an optimization,
    // since that same final witness should be found by evaluating a
    // LookupImplWitness instruction for the required self+interface pair.
    auto result_witness_id = FindFinalWitnessFromFacetValue(
        context, witness_sources, req_impl.self_facet_value,
        req_impl.specific_interface);
    if (result_witness_id.has_value()) {
      // Found a final witness, use it.
      result_witness_ids.push_back(result_witness_id);
      continue;
    }

    auto witness_const_id = EvalOrAddInst<SemIR::LookupImplWitness>(
        context, context.insts().GetLocIdForDesugaring(loc_id),
        {.type_id = GetSingletonType(context, SemIR::WitnessType::TypeInstId),
         .query_self_inst_id =
             context.constant_values().GetInstId(req_impl.self_facet_value),
         .query_specific_interface_id =
             context.specific_interfaces().Add(req_impl.specific_interface)});
    result_witness_id = context.constant_values().GetInstId(witness_const_id);
    if (!context.insts().Is<SemIR::LookupImplWitness>(result_witness_id)) {
      // Found a final witness, use it.
      result_witness_ids.push_back(result_witness_id);
      continue;
    }

    if (QueryIsConcrete(context, req_impl.self_facet_value,
                        req_impl.specific_interface)) {
      // Failed to find a final witness for a concrete query. There won't be a
      // non-final witness, as any witness would have been treated as final.
      break;
    }

    // Did not find a final witness. If we find a non-final witness, then we use
    // the `LookupImplWitness` as our witness so that monomorphization can
    // produce a final witness later.
    if (!FindNonFinalWitness(context, loc_id, witness_sources,
                             req_impl.self_facet_value,
                             req_impl.specific_interface)) {
      // At least one queried interface in the facet type has no witness for the
      // given type, we can stop looking for more.
      break;
    }

    // Save the non-final witness, which will eventually resolve to a final
    // witness as specifics are applied to make the query more concrete.
    result_witness_ids.push_back(result_witness_id);
  }
  auto pop = stack.pop_back_val();
  if (pop.diagnosed_cycle && !stack.empty()) {
    stack.back().diagnosed_cycle = true;
  }

  // All interfaces in the query facet type must have been found to be available
  // through some impl, or directly on the value's facet type if
  // `query_self_const_id` is a facet value.
  if (result_witness_ids.size() != req_impls.size()) {
    return SemIR::InstBlockId::None;
  }

  // Verify rewrite constraints in the query constraint are satisfied after
  // applying the rewrites from the found witnesses.
  if (!VerifyQueryFacetTypeConstraints(context, loc_id, query_self_const_id,
                                       query_facet_type_const_id, req_impls,
                                       result_witness_ids)) {
    return SemIR::InstBlockId::None;
  }

  return context.inst_blocks().AddCanonical(result_witness_ids);
}

auto GetCanonicalQuerySelfForLookupImplWitness(Context& context,
                                               SemIR::ConstantId self,
                                               SemIR::InstId* out_facet_value)
    -> SemIR::ConstantId {
  auto self_inst_id = context.constant_values().GetInstId(self);

  // If the monomorphized query self is a FacetValue, we may get a witness from
  // it under limited circumstances. If no final witness is found though, we
  // don't need to preserve it for future evaluations, so we strip it from the
  // LookupImplWitness instruction to reduce the number of distinct constant
  // values.
  if (auto facet_value =
          context.insts().TryGetAs<SemIR::FacetValue>(self_inst_id)) {
    if (out_facet_value) {
      *out_facet_value = self_inst_id;
    }
    self_inst_id = facet_value->type_inst_id;
  }

  // The self value is canonicalized in order to produce a canonical
  // LookupImplWitness instruction, avoiding multiple constant values for
  // `<facet value>` and `<facet value> as type`, which always have the same
  // lookup result.
  return GetCanonicalFacetOrTypeValue(
      context, context.constant_values().Get(self_inst_id));
}

// Record the query which found a final impl witness. It's illegal to
// write a final impl afterward that would match the same query.
static auto PoisonImplLookupQuery(Context& context, SemIR::LocId loc_id,
                                  EvalImplLookupMode mode,
                                  SemIR::LookupImplWitness eval_query,
                                  SemIR::ConstantId witness_id,
                                  const SemIR::Impl& impl) -> void {
  if (mode == EvalImplLookupMode::RecheckPoisonedLookup) {
    return;
  }
  // If the impl was effectively final, then we don't need to poison here. A
  // change of query result will already be diagnosed at the point where the
  // new impl decl was written that changes the result.
  if (TreatImplAsFinal(context, impl)) {
    return;
  }
  context.poisoned_concrete_impl_lookup_queries().push_back(
      {.loc_id = loc_id, .query = eval_query, .witness_id = witness_id});
}

// Return whether the `FacetType` in `type_id` extends a single interface, and
// that it matches `specific_interface`.
static auto FacetTypeIsSingleInterface(
    Context& context, SemIR::TypeId type_id,
    SemIR::SpecificInterface specific_interface) -> bool {
  auto facet_type = context.types().GetAs<SemIR::FacetType>(type_id);
  const auto& facet_type_info =
      context.facet_types().Get(facet_type.facet_type_id);
  if (auto single = facet_type_info.TryAsSingleExtend()) {
    if (auto* si = std::get_if<SemIR::SpecificInterface>(&*single)) {
      return *si == specific_interface;
    }
  }
  return false;
}

auto EvalLookupSingleFinalWitness(Context& context, SemIR::LocId loc_id,
                                  SemIR::LookupImplWitness eval_query,
                                  SemIR::InstId self_facet_value_inst_id,
                                  EvalImplLookupMode mode)
    -> SemIR::ConstantId {
  auto query_specific_interface =
      context.specific_interfaces().Get(eval_query.query_specific_interface_id);

  // Ensure specifics don't substitute in weird things for the query self.
  CARBON_CHECK(context.types().IsFacetType(
      context.insts().Get(eval_query.query_self_inst_id).type_id()));
  SemIR::ConstantId query_self_const_id =
      context.constant_values().Get(eval_query.query_self_inst_id);

  // If the query self is monomorphized as a FacetValue, we can't use its
  // witnesses in general, since we are not allowed to identify facet types in
  // monomorphization. And we need to identify it to know which witness is for
  // which interface.
  //
  // However, if the facet type has only a single interface and it matches the
  // query, then we can use the witness, since there is only one.
  //
  // This looks like an optimization, but it's done to prefer the FacetValue's
  // witness over the cached value for monomorphizations of `Self` inside an
  // `impl` definition. If a final witness was previously found for the same
  // type as the monomorphized `Self`, the cache would reuse it. But associated
  // constants may differ in that witness from the current `impl`'s witness
  // which leads to inconsistency within the impl definition.
  //
  // By preferring the impl's FacetValue, the `impl` remains self-consistent
  // even if it's ultimately not valid due to a conflict. When a conflict with
  // another `impl` does exist, a poisoning error will occur showing the two
  // `impl`s are in disagreement for a concrete value, as the poisoning lookup
  // does not preserve the FacetValue.
  if (auto facet_value = context.insts().TryGetAsIfValid<SemIR::FacetValue>(
          self_facet_value_inst_id)) {
    if (FacetTypeIsSingleInterface(context, facet_value->type_id,
                                   query_specific_interface)) {
      auto witnesses =
          context.inst_blocks().Get(facet_value->witnesses_block_id);
      CARBON_CHECK(witnesses.size() == 1);
      auto witness_inst_id = witnesses.front();
      // Only use the witness in monomoprhization if it's a final witness.
      if (!context.insts().Is<SemIR::LookupImplWitness>(witness_inst_id)) {
        return context.constant_values().Get(witness_inst_id);
      }
    }
  }

  // If the query is on `.Self` and looking for the same interface as `.Self`
  // provides, do not look for a witness in monomorphization - a non-final
  // witness will be found from the facet type. This happens inside an `impl`
  // declaration, and we must avoid finding that same `impl` and trying to
  // deduce `.Self` for it, as that results in a specific declaration for the
  // `impl` which evaluates this lookup again, producing a cycle.
  //
  // If the query is for `.Self` and for the facet type of `.Self`, then there
  // is no final witness yet.
  if (auto bind = context.insts().TryGetAs<SemIR::SymbolicBinding>(
          eval_query.query_self_inst_id)) {
    const auto& entity = context.entity_names().Get(bind->entity_name_id);
    if (entity.name_id == SemIR::NameId::PeriodSelf) {
      if (FacetTypeIsSingleInterface(context, bind->type_id,
                                     query_specific_interface)) {
        return SemIR::ConstantId::None;
      }
    }
  }

  // Check to see if this result is in the cache. But skip the cache if we're
  // re-checking a poisoned result and need to redo the lookup.
  auto impl_lookup_cache_key = Context::ImplLookupCacheKey{
      query_self_const_id, eval_query.query_specific_interface_id};
  if (mode != EvalImplLookupMode::RecheckPoisonedLookup) {
    if (auto result =
            context.impl_lookup_cache().Lookup(impl_lookup_cache_key)) {
      return result.value();
    }
  }

  bool query_is_concrete =
      QueryIsConcrete(context, query_self_const_id, query_specific_interface);

  auto query_type_structure = BuildTypeStructure(
      context, context.constant_values().GetInstId(query_self_const_id),
      query_specific_interface);
  if (!query_type_structure) {
    // TODO: We should return an error here; an error was found in the type
    // structure.
    return SemIR::ConstantId::None;
  }

  // We only want to return final witneses in monomorphization. If the query is
  // concrete, we can find all impls, otherwise we want only (effectively) final
  // impls.
  auto candidates = CollectCandidateImplsForQuery(
      context, /*final_only=*/!query_is_concrete, query_self_const_id,
      *query_type_structure, query_specific_interface);

  struct LookupResult {
    SemIR::ConstantId witness_id = SemIR::ConstantId::None;
    // Holds a pointer into `candidates`.
    const TypeStructure* impl_type_structure = nullptr;
    SemIR::LocId impl_loc_id = SemIR::LocId::None;
  };

  LookupResult lookup_result;

  auto core_interface =
      GetCoreInterface(context, query_specific_interface.interface_id);

  // Consider a custom witness for core interfaces.
  // TODO: This needs to expand to more interfaces, and we might want to have
  // that dispatch in custom_witness.cpp instead of here.
  bool used_custom_witness = false;
  if (auto witness_inst_id = LookupCustomWitness(
          context, loc_id, core_interface, query_self_const_id,
          eval_query.query_specific_interface_id, true)) {
    if (witness_inst_id->has_value()) {
      lookup_result = {.witness_id =
                           context.constant_values().Get(*witness_inst_id)};
      used_custom_witness = true;
    }
  }

  // Only consider candidates when a custom witness didn't apply.
  if (!used_custom_witness) {
    for (const auto& candidate : candidates.impls) {
      const auto& impl = *candidate.impl;

      // In monomorphization, while resolving a specific, there may be no stack
      // yet as this may be the first lookup. If further lookups are started as
      // a result in deduce, they will build the stack.
      if (!context.impl_lookup_stack().empty()) {
        context.impl_lookup_stack().back().impl_loc = impl.definition_id;
      }

      auto witness_id = TryGetSpecificWitnessIdForImpl(
          context, loc_id, query_self_const_id, query_specific_interface, impl);
      if (witness_id.has_value()) {
        PoisonImplLookupQuery(context, loc_id, mode, eval_query, witness_id,
                              impl);
        lookup_result = {.witness_id = witness_id,
                         .impl_type_structure = &candidate.type_structure,
                         .impl_loc_id = SemIR::LocId(impl.definition_id)};
        break;
      }
    }
  }

  if (query_is_concrete && candidates.consider_cpp_candidates &&
      core_interface != SemIR::CoreInterface::Unknown) {
    // Also check for a C++ candidate that is a better match than whatever
    // `impl` we may have found in Carbon.
    auto cpp_witness_id = LookupCppImpl(
        context, loc_id, core_interface, query_self_const_id,
        eval_query.query_specific_interface_id,
        lookup_result.impl_type_structure, lookup_result.impl_loc_id);
    if (cpp_witness_id.has_value()) {
      lookup_result = {.witness_id =
                           context.constant_values().Get(cpp_witness_id)};
    }
  }

  if (mode != EvalImplLookupMode::RecheckPoisonedLookup &&
      lookup_result.witness_id.has_value()) {
    context.impl_lookup_cache().Insert(impl_lookup_cache_key,
                                       lookup_result.witness_id);
  }
  return lookup_result.witness_id;
}

auto LookupMatchesImpl(Context& context, SemIR::LocId loc_id,
                       SemIR::ConstantId query_self_const_id,
                       SemIR::SpecificInterface query_specific_interface,
                       SemIR::ImplId target_impl) -> bool {
  if (query_self_const_id == SemIR::ErrorInst::ConstantId) {
    return false;
  }
  auto witness_id = TryGetSpecificWitnessIdForImpl(
      context, loc_id, query_self_const_id, query_specific_interface,
      context.impls().Get(target_impl));
  // TODO: If this fails, it would be because there is an error in the specific
  // interface. Should we check for that and return false?
  CARBON_CHECK(witness_id != SemIR::ErrorInst::ConstantId,
               "error in lookup specific interface");
  return witness_id.has_value();
}

}  // namespace Carbon::Check
