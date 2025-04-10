// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/facet_type.h"

#include "toolchain/check/convert.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/interface.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto FacetTypeFromInterface(Context& context, SemIR::InterfaceId interface_id,
                            SemIR::SpecificId specific_id) -> SemIR::FacetType {
  SemIR::FacetTypeId facet_type_id = context.facet_types().Add(
      SemIR::FacetTypeInfo{.extend_constraints = {{interface_id, specific_id}},
                           .other_requirements = false});
  return {.type_id = SemIR::TypeType::SingletonTypeId,
          .facet_type_id = facet_type_id};
}

// Returns `true` if the `FacetAccessWitness` of `witness_id` matches
// `interface`.
static auto WitnessAccessMatchesInterface(
    Context& context, SemIR::InstId witness_id,
    const SemIR::SpecificInterface& interface) -> bool {
  auto access = context.insts().GetAs<SemIR::FacetAccessWitness>(witness_id);
  auto type_id = context.insts().Get(access.facet_value_inst_id).type_id();
  auto facet_type = context.types().GetAs<SemIR::FacetType>(type_id);
  // The order of witnesses is from the identified facet type.
  auto identified_id = RequireIdentifiedFacetType(context, facet_type);
  const auto& identified = context.identified_facet_types().Get(identified_id);
  const auto& impls = identified.required_interfaces()[access.index.index];
  return impls == interface;
}

static auto IncompleteFacetTypeDiagnosticBuilder(
    Context& context, SemIRLoc loc, SemIR::InstId facet_type_inst_id,
    bool is_definition) -> DiagnosticBuilder {
  if (is_definition) {
    CARBON_DIAGNOSTIC(ImplAsIncompleteFacetTypeDefinition, Error,
                      "definition of impl as incomplete facet type {0}",
                      InstIdAsType);
    return context.emitter().Build(loc, ImplAsIncompleteFacetTypeDefinition,
                                   facet_type_inst_id);
  } else {
    CARBON_DIAGNOSTIC(
        ImplAsIncompleteFacetTypeRewrites, Error,
        "declaration of impl as incomplete facet type {0} with rewrites",
        InstIdAsType);
    return context.emitter().Build(loc, ImplAsIncompleteFacetTypeRewrites,
                                   facet_type_inst_id);
  }
}

auto InitialFacetTypeImplWitness(
    Context& context, SemIR::LocId witness_loc_id,
    SemIR::InstId facet_type_inst_id, SemIR::InstId self_type_inst_id,
    const SemIR::SpecificInterface& interface_to_witness,
    SemIR::SpecificId self_specific_id, bool is_definition) -> SemIR::InstId {
  // TODO: Finish facet type resolution. This code currently only handles
  // rewrite constraints that set associated constants to a concrete value.
  // Need logic to topologically sort rewrites to respect dependencies, and
  // afterwards reject duplicates that are not identical.

  auto facet_type_id =
      context.types().GetTypeIdForTypeInstId(facet_type_inst_id);
  CARBON_CHECK(facet_type_id != SemIR::ErrorInst::SingletonTypeId);
  auto facet_type = context.types().GetAs<SemIR::FacetType>(facet_type_id);
  // TODO: This is currently a copy because I'm not sure whether anything could
  // cause the facet type store to resize before we are done with it.
  auto facet_type_info = context.facet_types().Get(facet_type.facet_type_id);

  if (!is_definition && facet_type_info.rewrite_constraints.empty()) {
    return AddInst<SemIR::ImplWitness>(
        context, witness_loc_id,
        {.type_id =
             GetSingletonType(context, SemIR::WitnessType::SingletonInstId),
         .elements_id = context.inst_blocks().AddPlaceholder(),
         .specific_id = self_specific_id});
  }

  if (!RequireCompleteType(context, facet_type_id,
                           context.insts().GetLocId(facet_type_inst_id), [&] {
                             return IncompleteFacetTypeDiagnosticBuilder(
                                 context, witness_loc_id, facet_type_inst_id,
                                 is_definition);
                           })) {
    return SemIR::ErrorInst::SingletonInstId;
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
    llvm::SmallVector<SemIR::InstId> empty_table(
        assoc_entities.size(),
        SemIR::ImplWitnessTablePlaceholder::SingletonInstId);
    auto table_id = context.inst_blocks().Add(empty_table);
    table = context.inst_blocks().GetMutable(table_id);
    witness_inst_id = AddInst<SemIR::ImplWitness>(
        context, witness_loc_id,
        {.type_id =
             GetSingletonType(context, SemIR::WitnessType::SingletonInstId),
         .elements_id = table_id,
         .specific_id = self_specific_id});
  }

  for (auto rewrite : facet_type_info.rewrite_constraints) {
    auto access =
        context.insts().GetAs<SemIR::ImplWitnessAccess>(rewrite.lhs_id);
    if (!WitnessAccessMatchesInterface(context, access.witness_id,
                                       interface_to_witness)) {
      continue;
    }
    auto& table_entry = table[access.index.index];
    if (table_entry == SemIR::ErrorInst::SingletonInstId) {
      // Don't overwrite an error value. This prioritizes not generating
      // multiple errors for one associated constant over picking a value
      // for it to use to attempt recovery.
      continue;
    }
    auto rewrite_value = rewrite.rhs_id;
    if (rewrite_value == SemIR::ErrorInst::SingletonInstId) {
      table_entry = SemIR::ErrorInst::SingletonInstId;
      continue;
    }

    auto decl_id = context.constant_values().GetConstantInstId(
        assoc_entities[access.index.index]);
    CARBON_CHECK(decl_id.has_value(), "Non-constant associated entity");
    if (decl_id == SemIR::ErrorInst::SingletonInstId) {
      table_entry = SemIR::ErrorInst::SingletonInstId;
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
      table_entry = SemIR::ErrorInst::SingletonInstId;
      continue;
    }

    if (table_entry != SemIR::ImplWitnessTablePlaceholder::SingletonInstId) {
      if (table_entry != rewrite_value) {
        // TODO: Figure out how to print the two different values
        // `const_id` & `rewrite_value` in the diagnostic
        // message.
        CARBON_DIAGNOSTIC(AssociatedConstantWithDifferentValues, Error,
                          "associated constant {0} given two different values",
                          SemIR::NameId);
        auto& assoc_const = context.associated_constants().Get(
            assoc_constant_decl->assoc_const_id);
        context.emitter().Emit(facet_type_inst_id,
                               AssociatedConstantWithDifferentValues,
                               assoc_const.name_id);
      }
      table_entry = SemIR::ErrorInst::SingletonInstId;
      continue;
    }

    // If the associated constant has a symbolic type, convert the rewrite
    // value to that type now we know the value of `Self`.
    SemIR::TypeId assoc_const_type_id = assoc_constant_decl->type_id;
    if (assoc_const_type_id.is_symbolic()) {
      // Get the type of the associated constant in this interface with this
      // value for `Self`.
      assoc_const_type_id = GetTypeForSpecificAssociatedEntity(
          context, facet_type_inst_id, interface_to_witness.specific_id,
          decl_id, context.types().GetTypeIdForTypeInstId(self_type_inst_id),
          witness_inst_id);
      // Perform the conversion of the value to the type. We skipped this when
      // forming the facet type because the type of the associated constant
      // was symbolic.
      auto converted_inst_id = ConvertToValueOfType(
          context, context.insts().GetLocId(facet_type_inst_id), rewrite_value,
          assoc_const_type_id);
      // Canonicalize the converted constant value.
      converted_inst_id =
          context.constant_values().GetConstantInstId(converted_inst_id);
      // The result of conversion can be non-constant even if the original
      // value was constant.
      if (converted_inst_id.has_value()) {
        rewrite_value = converted_inst_id;
      } else if (rewrite_value != SemIR::ErrorInst::SingletonInstId) {
        const auto& assoc_const = context.associated_constants().Get(
            assoc_constant_decl->assoc_const_id);
        CARBON_DIAGNOSTIC(
            AssociatedConstantNotConstantAfterConversion, Error,
            "associated constant {0} given value that is not constant "
            "after conversion to {1}",
            SemIR::NameId, SemIR::TypeId);
        context.emitter().Emit(facet_type_inst_id,
                               AssociatedConstantNotConstantAfterConversion,
                               assoc_const.name_id, assoc_const_type_id);
        rewrite_value = SemIR::ErrorInst::SingletonInstId;
      }
    }

    CARBON_CHECK(rewrite_value ==
                     context.constant_values().GetConstantInstId(rewrite_value),
                 "Rewritten value for associated constant is not canonical.");

    table_entry = AddInst<SemIR::ImplWitnessAssociatedConstant>(
        context, witness_loc_id,
        {.type_id = context.insts().Get(rewrite_value).type_id(),
         .inst_id = rewrite_value});
  }
  return witness_inst_id;
}

auto RequireCompleteFacetTypeForImplDefinition(Context& context, SemIRLoc loc,
                                               SemIR::InstId facet_type_inst_id)
    -> bool {
  auto facet_type_id =
      context.types().GetTypeIdForTypeInstId(facet_type_inst_id);
  return RequireCompleteType(context, facet_type_id,
                             context.insts().GetLocId(facet_type_inst_id), [&] {
                               return IncompleteFacetTypeDiagnosticBuilder(
                                   context, loc, facet_type_inst_id,
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
      assoc_entities.size(),
      SemIR::ImplWitnessTablePlaceholder::SingletonInstId);
  context.inst_blocks().ReplacePlaceholder(witness_id, empty_table);
}

}  // namespace Carbon::Check
