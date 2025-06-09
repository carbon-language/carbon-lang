// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/facet_type.h"

#include "toolchain/check/convert.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/interface.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto FacetTypeFromInterface(Context& context, SemIR::InterfaceId interface_id,
                            SemIR::SpecificId specific_id) -> SemIR::FacetType {
  SemIR::FacetTypeId facet_type_id = context.facet_types().Add(
      SemIR::FacetTypeInfo{.extend_constraints = {{interface_id, specific_id}},
                           .other_requirements = false});
  return {.type_id = SemIR::TypeType::TypeId, .facet_type_id = facet_type_id};
}

// Returns whether the `LookupImplWitness` of `witness_id` matches `interface`.
static auto WitnessQueryMatchesInterface(
    Context& context, SemIR::InstId witness_id,
    const SemIR::SpecificInterface& interface) -> bool {
  auto lookup = context.insts().GetAs<SemIR::LookupImplWitness>(witness_id);
  return interface ==
         context.specific_interfaces().Get(lookup.query_specific_interface_id);
}

static auto IncompleteFacetTypeDiagnosticBuilder(
    Context& context, SemIR::LocId loc_id, SemIR::TypeInstId facet_type_inst_id,
    bool is_definition) -> DiagnosticBuilder {
  if (is_definition) {
    CARBON_DIAGNOSTIC(ImplAsIncompleteFacetTypeDefinition, Error,
                      "definition of impl as incomplete facet type {0}",
                      InstIdAsType);
    return context.emitter().Build(loc_id, ImplAsIncompleteFacetTypeDefinition,
                                   facet_type_inst_id);
  } else {
    CARBON_DIAGNOSTIC(
        ImplAsIncompleteFacetTypeRewrites, Error,
        "declaration of impl as incomplete facet type {0} with rewrites",
        InstIdAsType);
    return context.emitter().Build(loc_id, ImplAsIncompleteFacetTypeRewrites,
                                   facet_type_inst_id);
  }
}

auto InitialFacetTypeImplWitness(
    Context& context, SemIR::LocId witness_loc_id,
    SemIR::TypeInstId facet_type_inst_id, SemIR::TypeInstId self_type_inst_id,
    const SemIR::SpecificInterface& interface_to_witness,
    SemIR::SpecificId self_specific_id, bool is_definition) -> SemIR::InstId {
  // TODO: Finish facet type resolution. This code currently only handles
  // rewrite constraints that set associated constants to a concrete value.
  // Need logic to topologically sort rewrites to respect dependencies, and
  // afterwards reject duplicates that are not identical.

  auto facet_type_id =
      context.types().GetTypeIdForTypeInstId(facet_type_inst_id);
  CARBON_CHECK(facet_type_id != SemIR::ErrorInst::TypeId);
  auto facet_type = context.types().GetAs<SemIR::FacetType>(facet_type_id);
  // TODO: This is currently a copy because I'm not sure whether anything could
  // cause the facet type store to resize before we are done with it.
  auto facet_type_info = context.facet_types().Get(facet_type.facet_type_id);

  if (!is_definition && facet_type_info.rewrite_constraints.empty()) {
    auto witness_table_inst_id = AddInst<SemIR::ImplWitnessTable>(
        context, witness_loc_id,
        {.elements_id = context.inst_blocks().AddPlaceholder(),
         .impl_id = SemIR::ImplId::None});
    return AddInst<SemIR::ImplWitness>(
        context, witness_loc_id,
        {.type_id = GetSingletonType(context, SemIR::WitnessType::TypeInstId),
         .witness_table_id = witness_table_inst_id,
         .specific_id = self_specific_id});
  }

  if (!RequireCompleteType(
          context, facet_type_id, SemIR::LocId(facet_type_inst_id), [&] {
            return IncompleteFacetTypeDiagnosticBuilder(
                context, witness_loc_id, facet_type_inst_id, is_definition);
          })) {
    return SemIR::ErrorInst::InstId;
  }

  const auto& interface =
      context.interfaces().Get(interface_to_witness.interface_id);
  auto assoc_entities =
      context.inst_blocks().Get(interface.associated_entities_id);
  // TODO: When this function is used for things other than just impls, may want
  // to only load the specific associated entities that are mentioned in rewrite
  // rules.
  for (auto decl_id : assoc_entities) {
    LoadImportRef(context, decl_id);
  }

  SemIR::InstId witness_inst_id = SemIR::InstId::None;
  llvm::MutableArrayRef<SemIR::InstId> table;
  {
    auto elements_id =
        context.inst_blocks().AddUninitialized(assoc_entities.size());
    table = context.inst_blocks().GetMutable(elements_id);
    for (auto& uninit : table) {
      uninit = SemIR::ImplWitnessTablePlaceholder::TypeInstId;
    }

    auto witness_table_inst_id = AddInst<SemIR::ImplWitnessTable>(
        context, witness_loc_id,
        {.elements_id = elements_id, .impl_id = SemIR::ImplId::None});

    witness_inst_id = AddInst<SemIR::ImplWitness>(
        context, witness_loc_id,
        {.type_id = GetSingletonType(context, SemIR::WitnessType::TypeInstId),
         .witness_table_id = witness_table_inst_id,
         .specific_id = self_specific_id});
  }

  for (auto rewrite : facet_type_info.rewrite_constraints) {
    auto access =
        context.insts().GetAs<SemIR::ImplWitnessAccess>(rewrite.lhs_id);
    if (!WitnessQueryMatchesInterface(context, access.witness_id,
                                      interface_to_witness)) {
      continue;
    }
    auto& table_entry = table[access.index.index];
    if (table_entry == SemIR::ErrorInst::InstId) {
      // Don't overwrite an error value. This prioritizes not generating
      // multiple errors for one associated constant over picking a value
      // for it to use to attempt recovery.
      continue;
    }
    auto rewrite_inst_id = rewrite.rhs_id;
    if (rewrite_inst_id == SemIR::ErrorInst::InstId) {
      table_entry = SemIR::ErrorInst::InstId;
      continue;
    }

    auto decl_id = context.constant_values().GetConstantInstId(
        assoc_entities[access.index.index]);
    CARBON_CHECK(decl_id.has_value(), "Non-constant associated entity");
    if (decl_id == SemIR::ErrorInst::InstId) {
      table_entry = SemIR::ErrorInst::InstId;
      continue;
    }

    auto assoc_constant_decl =
        context.insts().TryGetAs<SemIR::AssociatedConstantDecl>(decl_id);
    if (!assoc_constant_decl) {
      auto type_id = context.insts().Get(decl_id).type_id();
      auto type_inst = context.types().GetAsInst(type_id);
      auto fn_type = type_inst.As<SemIR::FunctionType>();
      const auto& fn = context.functions().Get(fn_type.function_id);
      CARBON_DIAGNOSTIC(RewriteForAssociatedFunction, Error,
                        "rewrite specified for associated function {0}",
                        SemIR::NameId);
      context.emitter().Emit(facet_type_inst_id, RewriteForAssociatedFunction,
                             fn.name_id);
      table_entry = SemIR::ErrorInst::InstId;
      continue;
    }

    // FacetTypes resolution disallows two rewrites to the same associated
    // constant, so we won't ever have a facet write twice to the same position
    // in the witness table.
    CARBON_CHECK(table_entry == SemIR::ImplWitnessTablePlaceholder::TypeInstId);

    // If the associated constant has a symbolic type, convert the rewrite
    // value to that type now we know the value of `Self`.
    SemIR::TypeId assoc_const_type_id = assoc_constant_decl->type_id;
    if (assoc_const_type_id.is_symbolic()) {
      // Get the type of the associated constant in this interface with this
      // value for `Self`.
      assoc_const_type_id = GetTypeForSpecificAssociatedEntity(
          context, SemIR::LocId(facet_type_inst_id),
          interface_to_witness.specific_id, decl_id,
          context.types().GetTypeIdForTypeInstId(self_type_inst_id),
          witness_inst_id);
      // Perform the conversion of the value to the type. We skipped this when
      // forming the facet type because the type of the associated constant
      // was symbolic.
      auto converted_inst_id =
          ConvertToValueOfType(context, SemIR::LocId(facet_type_inst_id),
                               rewrite_inst_id, assoc_const_type_id);
      // Canonicalize the converted constant value.
      converted_inst_id =
          context.constant_values().GetConstantInstId(converted_inst_id);
      // The result of conversion can be non-constant even if the original
      // value was constant.
      if (converted_inst_id.has_value()) {
        rewrite_inst_id = converted_inst_id;
      } else {
        const auto& assoc_const = context.associated_constants().Get(
            assoc_constant_decl->assoc_const_id);
        CARBON_DIAGNOSTIC(
            AssociatedConstantNotConstantAfterConversion, Error,
            "associated constant {0} given value {1} that is not constant "
            "after conversion to {2}",
            SemIR::NameId, InstIdAsConstant, SemIR::TypeId);
        context.emitter().Emit(
            facet_type_inst_id, AssociatedConstantNotConstantAfterConversion,
            assoc_const.name_id, rewrite_inst_id, assoc_const_type_id);
        rewrite_inst_id = SemIR::ErrorInst::InstId;
      }
    }

    CARBON_CHECK(rewrite_inst_id == context.constant_values().GetConstantInstId(
                                        rewrite_inst_id),
                 "Rewritten value for associated constant is not canonical.");

    table_entry = AddInst<SemIR::ImplWitnessAssociatedConstant>(
        context, witness_loc_id,
        {.type_id = context.insts().Get(rewrite_inst_id).type_id(),
         .inst_id = rewrite_inst_id});
  }
  return witness_inst_id;
}

auto RequireCompleteFacetTypeForImplDefinition(
    Context& context, SemIR::LocId loc_id, SemIR::TypeInstId facet_type_inst_id)
    -> bool {
  auto facet_type_id =
      context.types().GetTypeIdForTypeInstId(facet_type_inst_id);
  return RequireCompleteType(
      context, facet_type_id, SemIR::LocId(facet_type_inst_id), [&] {
        return IncompleteFacetTypeDiagnosticBuilder(context, loc_id,
                                                    facet_type_inst_id,
                                                    /*is_definition=*/true);
      });
}

auto AllocateFacetTypeImplWitness(Context& context,
                                  SemIR::InterfaceId interface_id,
                                  SemIR::InstBlockId witness_id) -> void {
  const auto& interface = context.interfaces().Get(interface_id);
  CARBON_CHECK(interface.is_complete());
  auto assoc_entities =
      context.inst_blocks().Get(interface.associated_entities_id);
  for (auto decl_id : assoc_entities) {
    LoadImportRef(context, decl_id);
  }

  llvm::SmallVector<SemIR::InstId> empty_table(
      assoc_entities.size(), SemIR::ImplWitnessTablePlaceholder::TypeInstId);
  context.inst_blocks().ReplacePlaceholder(witness_id, empty_table);
}

auto IsPeriodSelf(Context& context, SemIR::ConstantId const_id) -> bool {
  // This also rejects the singleton Error value as it's concrete.
  if (!const_id.is_symbolic()) {
    return false;
  }
  const auto& symbolic =
      context.constant_values().GetSymbolicConstant(const_id);
  // Fast early reject before doing more expensive operations.
  if (symbolic.dependence != SemIR::ConstantDependence::PeriodSelf) {
    return false;
  }
  return IsPeriodSelf(context, symbolic.inst_id);
}

auto IsPeriodSelf(Context& context, SemIR::InstId inst_id) -> bool {
  // Unwrap the `FacetAccessType` instruction, which we get when the `.Self` is
  // converted to `type`.
  if (auto facet_access_type =
          context.insts().TryGetAs<SemIR::FacetAccessType>(inst_id)) {
    inst_id = facet_access_type->facet_value_inst_id;
  }
  if (auto bind_symbolic_name =
          context.insts().TryGetAs<SemIR::BindSymbolicName>(inst_id)) {
    const auto& bind_name =
        context.entity_names().Get(bind_symbolic_name->entity_name_id);
    return bind_name.name_id == SemIR::NameId::PeriodSelf;
  }
  return false;
}

auto ResolveRewriteConstraintsAndCanonicalize(Context& context,
                                              SemIR::LocId loc_id,
                                              SemIR::FacetTypeInfo& facet_type)
    -> void {
  // This operation sorts and dedupes the rewrite constraints. They are sorted
  // primarily by the `lhs_id`, then by the `rhs_id`.
  facet_type.Canonicalize();

  if (facet_type.rewrite_constraints.empty()) {
    return;
  }

  for (size_t i = 0; i < facet_type.rewrite_constraints.size() - 1; ++i) {
    auto& constraint = facet_type.rewrite_constraints[i];
    if (constraint.lhs_id == SemIR::ErrorInst::InstId ||
        constraint.rhs_id == SemIR::ErrorInst::InstId) {
      continue;
    }

    auto lhs_access =
        context.insts().TryGetAs<SemIR::ImplWitnessAccess>(constraint.lhs_id);
    if (!lhs_access) {
      continue;
    }
    auto lhs_lookup = context.insts().TryGetAs<SemIR::LookupImplWitness>(
        lhs_access->witness_id);
    if (!lhs_lookup) {
      continue;
    }
    if (!IsPeriodSelf(context, lhs_lookup->query_self_inst_id)) {
      continue;
    }

    // This loop moves `i` to the last position with the same LHS value, so that
    // we don't diagnose more than once within the same contiguous range of
    // assignments to a single LHS value.
    for (; i < facet_type.rewrite_constraints.size() - 1; ++i) {
      auto& next = facet_type.rewrite_constraints[i + 1];
      if (constraint.lhs_id != next.lhs_id) {
        break;
      }
      // `constraint.lhs_id == next.lhs_id` so only check for `ErrorInst` in the
      // RHS. On the first error, `constraint.rhs_id` is set to `ErrorInst`
      // which prevents further diagnostics for the same LHS value due to this
      // condition.
      if (constraint.rhs_id != SemIR::ErrorInst::InstId &&
          next.rhs_id != SemIR::ErrorInst::InstId) {
        CARBON_DIAGNOSTIC(
            AssociatedConstantWithDifferentValues, Error,
            "associated constant {0} given two different values {1} and {2}",
            InstIdAsConstant, InstIdAsConstant, InstIdAsConstant);
        // TODO: It would be nice to note the places where the values are
        // assigned but rewrite constraint instructions are from canonical
        // constant values, and have no locations. We'd need to store a location
        // along with them in the rewrite constraints.
        context.emitter().Emit(loc_id, AssociatedConstantWithDifferentValues,
                               constraint.lhs_id, constraint.rhs_id,
                               next.rhs_id);
      }
      constraint.rhs_id = SemIR::ErrorInst::InstId;
      next.rhs_id = SemIR::ErrorInst::InstId;
    }
  }

  // Canonicalize again, as we may have inserted errors into the rewrite
  // constraints, and these could change sorting order.
  facet_type.Canonicalize();
}

}  // namespace Carbon::Check
