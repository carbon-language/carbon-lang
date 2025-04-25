// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/impl_lookup.h"

#include <algorithm>
#include <functional>
#include <utility>
#include <variant>

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/deduce.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/impl.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/check/type_structure.h"
#include "toolchain/sem_ir/facet_type_info.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/impl.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

static auto FindAssociatedImportIRs(Context& context,
                                    SemIR::ConstantId query_self_const_id,
                                    SemIR::ConstantId query_facet_type_const_id)
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
    if (auto ir_id = GetCanonicalImportIRInst(context, decl_id).ir_id();
        ir_id.has_value()) {
      result.push_back(ir_id);
    }
  };

  llvm::SmallVector<SemIR::InstId> worklist;
  worklist.push_back(context.constant_values().GetInstId(query_self_const_id));
  if (query_facet_type_const_id.has_value()) {
    worklist.push_back(
        context.constant_values().GetInstId(query_facet_type_const_id));
  }

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
    Context& context,
    const llvm::SmallVector<Context::ImplLookupStackEntry>& stack,
    SemIR::LocId loc_id, SemIR::ConstantId query_self_const_id,
    SemIR::ConstantId query_facet_type_const_id) -> bool {
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
      auto facet_type_type_id =
          context.types().GetTypeIdForTypeConstantId(query_facet_type_const_id);
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
      return true;
    }
  }
  return false;
}

// Gets the set of `SpecificInterface`s that are required by a facet type
// (as a constant value).
static auto GetInterfacesFromConstantId(
    Context& context, SemIR::ConstantId query_facet_type_const_id,
    bool& has_other_requirements)
    -> llvm::SmallVector<SemIR::SpecificInterface> {
  auto facet_type_inst_id =
      context.constant_values().GetInstId(query_facet_type_const_id);
  auto facet_type_inst =
      context.insts().GetAs<SemIR::FacetType>(facet_type_inst_id);
  const auto& facet_type_info =
      context.facet_types().Get(facet_type_inst.facet_type_id);
  has_other_requirements = facet_type_info.other_requirements;
  auto identified_id = RequireIdentifiedFacetType(context, facet_type_inst);
  auto interfaces_array_ref =
      context.identified_facet_types().Get(identified_id).required_interfaces();
  // Returns a copy to avoid use-after-free when the identified_facet_types
  // store resizes.
  return {interfaces_array_ref.begin(), interfaces_array_ref.end()};
}

static auto GetWitnessIdForImpl(Context& context, SemIR::LocId loc_id,
                                bool query_is_concrete,
                                SemIR::ConstantId query_self_const_id,
                                const SemIR::SpecificInterface& interface,
                                SemIR::ImplId impl_id) -> EvalImplLookupResult {
  // The impl may have generic arguments, in which case we need to deduce them
  // to find what they are given the specific type and interface query. We use
  // that specific to map values in the impl to the deduced values.
  auto specific_id = SemIR::SpecificId::None;
  {
    // DeduceImplArguments can import new impls which can invalidate any
    // pointers into `context.impls()`.
    const SemIR::Impl& impl = context.impls().Get(impl_id);

    if (impl.generic_id.has_value()) {
      specific_id =
          DeduceImplArguments(context, loc_id,
                              {.self_id = impl.self_id,
                               .generic_id = impl.generic_id,
                               .specific_id = impl.interface.specific_id},
                              query_self_const_id, interface.specific_id);
      if (!specific_id.has_value()) {
        return EvalImplLookupResult::MakeNone();
      }
    }
  }

  // Get a pointer again after DeduceImplArguments() is complete.
  const SemIR::Impl& impl = context.impls().Get(impl_id);

  // The self type of the impl must match the type in the query, or this is an
  // `impl T as ...` for some other type `T` and should not be considered.
  auto deduced_self_const_id = SemIR::GetConstantValueInSpecific(
      context.sem_ir(), specific_id, impl.self_id);
  // In a generic `impl forall` the self type can be a FacetAccessType, which
  // will not be the same constant value as a query facet value. We move through
  // to the facet value here, and if the query was a FacetAccessType we did the
  // same there so they still match.
  deduced_self_const_id =
      GetCanonicalizedFacetOrTypeValue(context, deduced_self_const_id);
  if (query_self_const_id != deduced_self_const_id) {
    return EvalImplLookupResult::MakeNone();
  }

  // The impl's constraint is a facet type which it is implementing for the self
  // type: the `I` in `impl ... as I`. The deduction step may be unable to be
  // fully applied to the types in the constraint and result in an error here,
  // in which case it does not match the query.
  auto deduced_constraint_id =
      context.constant_values().GetInstId(SemIR::GetConstantValueInSpecific(
          context.sem_ir(), specific_id, impl.constraint_id));
  if (deduced_constraint_id == SemIR::ErrorInst::InstId) {
    return EvalImplLookupResult::MakeNone();
  }

  auto deduced_constraint_facet_type_id =
      context.insts()
          .GetAs<SemIR::FacetType>(deduced_constraint_id)
          .facet_type_id;
  const auto& deduced_constraint_facet_type_info =
      context.facet_types().Get(deduced_constraint_facet_type_id);
  CARBON_CHECK(deduced_constraint_facet_type_info.extend_constraints.size() ==
               1);

  if (deduced_constraint_facet_type_info.other_requirements) {
    // TODO: Remove this when other requirements goes away.
    return EvalImplLookupResult::MakeNone();
  }

  // The specifics in the queried interface must match the deduced specifics in
  // the impl's constraint facet type.
  auto impl_interface_specific_id =
      deduced_constraint_facet_type_info.extend_constraints[0].specific_id;
  auto query_interface_specific_id = interface.specific_id;
  if (impl_interface_specific_id != query_interface_specific_id) {
    return EvalImplLookupResult::MakeNone();
  }

  LoadImportRef(context, impl.witness_id);
  if (specific_id.has_value()) {
    // We need a definition of the specific `impl` so we can access its
    // witness.
    ResolveSpecificDefinition(context, loc_id, specific_id);
  }

  if (query_is_concrete || IsImplEffectivelyFinal(context, impl)) {
    // TODO: These final results should be cached somehow. Positive (non-None)
    // results could be cached globally, as they can not change. But
    // negative results can change after a final impl is written, so
    // they can only be cached in a limited way, or the cache needs to
    // be invalidated by writing a final impl that would match.
    return EvalImplLookupResult::MakeFinal(
        context.constant_values().GetInstId(SemIR::GetConstantValueInSpecific(
            context.sem_ir(), specific_id, impl.witness_id)));
  } else {
    return EvalImplLookupResult::MakeNonFinal();
  }
}

// Unwraps a FacetAccessType to move from a value of type `TypeType` to a facet
// value of type `FacetType` if possible.
//
// Generally `GetCanonicalizedFacetOrTypeValue()` is what you want to call
// instead, as this only does part of that operation, potentially returning a
// non-canonical facet value.
static auto UnwrapFacetAccessType(Context& context, SemIR::InstId inst_id)
    -> SemIR::InstId {
  if (auto access = context.insts().TryGetAs<SemIR::FacetAccessType>(inst_id)) {
    return access->facet_value_inst_id;
  }
  return inst_id;
}

// Finds a lookup result from `query_self_inst_id` if it is a facet value that
// names the query interface in its facet type. Note that `query_self_inst_id`
// is allowed to be a non-canonical facet value in order to find a concrete
// witness, so it's not referenced as a constant value.
static auto LookupImplWitnessInSelfFacetValue(
    Context& context, SemIR::InstId query_self_inst_id,
    SemIR::SpecificInterface query_specific_interface) -> EvalImplLookupResult {
  // Unwrap FacetAccessType without getting the canonical facet value from the
  // self value, as we want to preserve the non-canonical `FacetValue`
  // instruction which can contain the concrete witness.
  query_self_inst_id = UnwrapFacetAccessType(context, query_self_inst_id);

  auto facet_type = context.types().TryGetAs<SemIR::FacetType>(
      context.insts().Get(query_self_inst_id).type_id());
  if (!facet_type) {
    return EvalImplLookupResult::MakeNone();
  }

  // The position of the interface in `required_interfaces()` is also the
  // position of the witness for that interface in `FacetValue`.
  auto identified_id = RequireIdentifiedFacetType(context, *facet_type);
  auto facet_type_required_interfaces =
      llvm::enumerate(context.identified_facet_types()
                          .Get(identified_id)
                          .required_interfaces());
  auto it = llvm::find_if(facet_type_required_interfaces, [=](auto e) {
    return e.value() == query_specific_interface;
  });
  if (it == facet_type_required_interfaces.end()) {
    return EvalImplLookupResult::MakeNone();
  }
  auto index = (*it).index();

  if (auto facet_value =
          context.insts().TryGetAs<SemIR::FacetValue>(query_self_inst_id)) {
    auto witness_id =
        context.inst_blocks().Get(facet_value->witnesses_block_id)[index];
    if (context.insts().Is<SemIR::ImplWitness>(witness_id)) {
      return EvalImplLookupResult::MakeFinal(witness_id);
    }
  }
  return EvalImplLookupResult::MakeNonFinal();
}

// Begin a search for an impl declaration matching the query. We do this by
// creating an LookupImplWitness instruction and evaluating. If it's able to
// find a final concrete impl, then it will evaluate to that `ImplWitness` but
// if not, it will evaluate to itself as a symbolic witness to be further
// evaluated with a more specific query when building a specific for the generic
// context the query came from.
static auto GetOrAddLookupImplWitness(Context& context, SemIR::LocId loc_id,
                                      SemIR::ConstantId query_self_const_id,
                                      SemIR::SpecificInterface interface)
    -> SemIR::InstId {
  auto witness_const_id = EvalOrAddInst(
      context, loc_id.ToImplicit(),
      SemIR::LookupImplWitness{
          .type_id = GetSingletonType(context, SemIR::WitnessType::TypeInstId),
          .query_self_inst_id =
              context.constant_values().GetInstId(query_self_const_id),
          .query_specific_interface_id =
              context.specific_interfaces().Add(interface),
      });
  // We use a NotConstant result from eval to communicate back an impl
  // lookup failure. See `EvalConstantInst()` for `LookupImplWitness`.
  if (!witness_const_id.is_constant()) {
    return SemIR::InstId::None;
  }
  return context.constant_values().GetInstId(witness_const_id);
}

auto LookupImplWitness(Context& context, SemIR::LocId loc_id,
                       SemIR::ConstantId query_self_const_id,
                       SemIR::ConstantId query_facet_type_const_id)
    -> SemIR::InstBlockIdOrError {
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
    CARBON_CHECK(context.types().Is<SemIR::TypeType>(query_self_type_id) ||
                 context.types().Is<SemIR::FacetType>(query_self_type_id));
    // The query facet type value is indeed a facet type.
    CARBON_CHECK(context.insts().Is<SemIR::FacetType>(
        context.constant_values().GetInstId(query_facet_type_const_id)));
  }

  auto import_irs = FindAssociatedImportIRs(context, query_self_const_id,
                                            query_facet_type_const_id);
  for (auto import_ir : import_irs) {
    // TODO: Instead of importing all impls, only import ones that are in some
    // way connected to this query.
    for (auto impl_index : llvm::seq(
             context.import_irs().Get(import_ir).sem_ir->impls().size())) {
      // TODO: Track the relevant impls and only consider those ones and any
      // local impls, rather than looping over all impls below.
      ImportImpl(context, import_ir, SemIR::ImplId(impl_index));
    }
  }

  if (FindAndDiagnoseImplLookupCycle(context, context.impl_lookup_stack(),
                                     loc_id, query_self_const_id,
                                     query_facet_type_const_id)) {
    return SemIR::InstBlockIdOrError::MakeError();
  }

  bool has_other_requirements = false;
  auto interfaces = GetInterfacesFromConstantId(
      context, query_facet_type_const_id, has_other_requirements);
  if (has_other_requirements) {
    // TODO: Remove this when other requirements go away.
    return SemIR::InstBlockId::None;
  }
  if (interfaces.empty()) {
    return SemIR::InstBlockId::Empty;
  }

  auto& stack = context.impl_lookup_stack();
  stack.push_back({
      .query_self_const_id = query_self_const_id,
      .query_facet_type_const_id = query_facet_type_const_id,
  });
  // We need to find a witness for each interface in `interfaces`. Every
  // consumer of a facet type needs to agree on the order of interfaces used for
  // its witnesses.
  llvm::SmallVector<SemIR::InstId> result_witness_ids;
  for (const auto& interface : interfaces) {
    // TODO: Since both `interfaces` and `query_self_const_id` are sorted lists,
    // do an O(N+M) merge instead of O(N*M) nested loops.
    auto result_witness_id = GetOrAddLookupImplWitness(
        context, loc_id, query_self_const_id, interface);
    if (result_witness_id.has_value()) {
      result_witness_ids.push_back(result_witness_id);
    } else {
      // At least one queried interface in the facet type has no witness for the
      // given type, we can stop looking for more.
      break;
    }
  }
  stack.pop_back();
  // TODO: Validate that the witness satisfies the other requirements in
  // `interface_const_id`.

  // All interfaces in the query facet type must have been found to be available
  // through some impl, or directly on the value's facet type if
  // `query_self_const_id` is a facet value.
  if (result_witness_ids.size() != interfaces.size()) {
    return SemIR::InstBlockId::None;
  }

  return context.inst_blocks().AddCanonical(result_witness_ids);
}

// Returns whether the query is concrete, it is false if the self type or
// interface specifics have a symbolic dependency.
static auto QueryIsConcrete(Context& context, SemIR::ConstantId self_const_id,
                            SemIR::SpecificInterface& specific_interface)
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

struct CandidateImpl {
  SemIR::ImplId impl_id;
  SemIR::InstId loc_inst_id;

  // Used for sorting the candidates to find the most-specialized match.
  TypeStructure type_structure;
};

// Returns the list of candidates impls for lookup to select from.
static auto CollectCandidateImplsForQuery(
    Context& context, bool final_only,
    const TypeStructure& query_type_structure,
    SemIR::SpecificInterface& query_specific_interface)
    -> llvm::SmallVector<CandidateImpl> {
  llvm::SmallVector<CandidateImpl> candidate_impls;
  for (auto [id, impl] : context.impls().enumerate()) {
    if (final_only && !IsImplEffectivelyFinal(context, impl)) {
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

    // This check comes first to avoid deduction with an invalid impl. We use
    // an error value to indicate an error during creation of the impl, such
    // as a recursive impl which will cause deduction to recurse infinitely.
    if (impl.witness_id == SemIR::ErrorInst::InstId) {
      continue;
    }
    CARBON_CHECK(impl.witness_id.has_value());

    // Build the type structure used for choosing the best the candidate.
    auto type_structure =
        BuildTypeStructure(context, impl.self_id, impl.interface);
    // TODO: We can skip the comparison here if the `impl_interface_const_id` is
    // not symbolic, since when the interface and specific ids match, and they
    // aren't symbolic, the structure will be identical.
    if (!query_type_structure.IsCompatibleWith(type_structure)) {
      continue;
    }

    candidate_impls.push_back(
        {id, impl.definition_id, std::move(type_structure)});
  }

  auto compare = [](auto& lhs, auto& rhs) -> bool {
    return lhs.type_structure < rhs.type_structure;
  };
  // Stable sort is used so that impls that are seen first are preferred when
  // they have an equal priority ordering.
  // TODO: Allow Carbon code to provide a priority ordering explicitly. For
  // now they have all the same priority, so the priority is the order in
  // which they are found in code.
  llvm::stable_sort(candidate_impls, compare);

  return candidate_impls;
}

auto EvalLookupSingleImplWitness(Context& context, SemIR::LocId loc_id,
                                 SemIR::LookupImplWitness eval_query,
                                 SemIR::InstId non_canonical_query_self_inst_id)
    -> EvalImplLookupResult {
  // NOTE: Do not retain this reference to the SpecificInterface obtained from a
  // value store by SpecificInterfaceId. Doing impl lookup does deduce which can
  // do more impl lookups, and impl lookup can add a new SpecificInterface to
  // the store which can reallocate and invalidate any references held here into
  // the store.
  auto query_specific_interface =
      context.specific_interfaces().Get(eval_query.query_specific_interface_id);

  auto facet_lookup_result = LookupImplWitnessInSelfFacetValue(
      context, non_canonical_query_self_inst_id, query_specific_interface);
  if (facet_lookup_result.has_concrete_value()) {
    return facet_lookup_result;
  }

  // If the self type is a facet that provides a witness, then we are in an
  // `interface` or an `impl`. In both cases, we don't want to do any impl
  // lookups. The query will eventually resolve to a concrete witness when it
  // can get it from the self facet value, when it has a specific applied in the
  // future.
  //
  // In particular, this avoids a LookupImplWitness instruction in the eval
  // block of an impl declaration from doing impl lookup. Specifically the
  // lookup of the implicit .Self in `impl ... where .X`. If it does impl lookup
  // when the eval block is run, it finds the same `impl`, tries to build a
  // specific from it, which runs the eval block, creating a recursive loop that
  // crashes.
  bool self_facet_provides_witness = facet_lookup_result.has_value();
  if (self_facet_provides_witness) {
    if (auto bind = context.insts().TryGetAs<SemIR::BindSymbolicName>(
            eval_query.query_self_inst_id)) {
      const auto& entity = context.entity_names().Get(bind->entity_name_id);
      if (entity.name_id == SemIR::NameId::PeriodSelf ||
          entity.name_id == SemIR::NameId::SelfType) {
        return EvalImplLookupResult::MakeNonFinal();
      }
    }
  }

  SemIR::ConstantId query_self_const_id =
      context.constant_values().Get(eval_query.query_self_inst_id);

  auto query_type_structure = BuildTypeStructure(
      context, context.constant_values().GetInstId(query_self_const_id),
      query_specific_interface);
  bool query_is_concrete =
      QueryIsConcrete(context, query_self_const_id, query_specific_interface);

  // If we have a symbolic witness in the self query, then the query can not be
  // concrete: the query includes a symbolic self value.
  CARBON_CHECK(!self_facet_provides_witness || !query_is_concrete);

  // If the self value is a (symbolic) facet value that has a symbolic witness,
  // then we don't need to do impl lookup, except that we want to find any final
  // impls to return a concrete witness if possible. So we limit the query to
  // final impls only in that case. Note as in the CHECK above, the query can
  // not be concrete in this case, so only final impls can produce a concrete
  // witness for this query.
  auto candidate_impls = CollectCandidateImplsForQuery(
      context, self_facet_provides_witness, query_type_structure,
      query_specific_interface);

  for (const auto& candidate : candidate_impls) {
    // In deferred lookup for a symbolic impl witness, while building a
    // specific, there may be no stack yet as this may be the first lookup. If
    // further lookups are started as a result in deduce, they will build the
    // stack.
    //
    // NOTE: Don't retain a reference into the stack, it may be invalidated if
    // we do further impl lookups when GetWitnessIdForImpl() does deduction.
    if (!context.impl_lookup_stack().empty()) {
      context.impl_lookup_stack().back().impl_loc = candidate.loc_inst_id;
    }

    // NOTE: GetWitnessIdForImpl() does deduction, which can cause new impls
    // to be imported, invalidating any pointer into `context.impls()`.
    auto result = GetWitnessIdForImpl(
        context, loc_id, query_is_concrete, query_self_const_id,
        query_specific_interface, candidate.impl_id);
    if (result.has_value()) {
      return result;
    }
  }

  if (self_facet_provides_witness) {
    // If we did not find a final impl, but the self value is a facet that
    // provides a symbolic witness, when we record that an impl will exist for
    // the specific, but is yet unknown.
    return EvalImplLookupResult::MakeNonFinal();
  }

  return EvalImplLookupResult::MakeNone();
}

}  // namespace Carbon::Check
