// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/impl.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/function.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/interface.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/impl.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Adds the location of the associated function to a diagnostic.
static auto NoteAssociatedFunction(Context& context,
                                   Context::DiagnosticBuilder& builder,
                                   SemIR::FunctionId function_id) -> void {
  CARBON_DIAGNOSTIC(AssociatedFunctionHere, Note,
                    "associated function {0} declared here", SemIR::NameId);
  const auto& function = context.functions().Get(function_id);
  builder.Note(function.latest_decl_id(), AssociatedFunctionHere,
               function.name_id);
}

// Checks that `impl_function_id` is a valid implementation of the function
// described in the interface as `interface_function_id`. Returns the value to
// put into the corresponding slot in the witness table, which can be
// `BuiltinErrorInst` if the function is not usable.
static auto CheckAssociatedFunctionImplementation(
    Context& context, SemIR::FunctionType interface_function_type,
    SemIR::InstId impl_decl_id, SemIR::TypeId self_type_id,
    SemIR::InstId witness_inst_id) -> SemIR::InstId {
  auto impl_function_decl =
      context.insts().TryGetAs<SemIR::FunctionDecl>(impl_decl_id);
  if (!impl_function_decl) {
    CARBON_DIAGNOSTIC(ImplFunctionWithNonFunction, Error,
                      "associated function {0} implemented by non-function",
                      SemIR::NameId);
    auto builder = context.emitter().Build(
        impl_decl_id, ImplFunctionWithNonFunction,
        context.functions().Get(interface_function_type.function_id).name_id);
    NoteAssociatedFunction(context, builder,
                           interface_function_type.function_id);
    builder.Emit();

    return SemIR::ErrorInst::SingletonInstId;
  }

  // Map from the specific for the function type to the specific for the
  // function signature. The function signature may have additional generic
  // parameters.
  auto interface_function_specific_id =
      GetSelfSpecificForInterfaceMemberWithSelfType(
          context, impl_decl_id, interface_function_type.specific_id,
          context.functions()
              .Get(interface_function_type.function_id)
              .generic_id,
          self_type_id, witness_inst_id);

  if (!CheckFunctionTypeMatches(
          context, context.functions().Get(impl_function_decl->function_id),
          context.functions().Get(interface_function_type.function_id),
          interface_function_specific_id,
          /*check_syntax=*/false)) {
    return SemIR::ErrorInst::SingletonInstId;
  }
  return impl_decl_id;
}

// Builds an initial empty witness.
// TODO: Fill the witness with the rewrites from the declaration.
auto ImplWitnessForDeclaration(Context& context, const SemIR::Impl& impl)
    -> SemIR::InstId {
  CARBON_CHECK(!impl.has_definition_started());

  auto self_type_id = context.types().GetTypeIdForTypeInstId(impl.self_id);
  if (self_type_id == SemIR::ErrorInst::SingletonTypeId) {
    // When 'impl as' is invalid, the self type is an error.
    return SemIR::ErrorInst::SingletonInstId;
  }
  auto facet_type_id =
      context.types().GetTypeIdForTypeInstId(impl.constraint_id);
  if (facet_type_id == SemIR::ErrorInst::SingletonTypeId) {
    return SemIR::ErrorInst::SingletonInstId;
  }
  auto facet_type = context.types().TryGetAs<SemIR::FacetType>(facet_type_id);
  if (!facet_type) {
    CARBON_DIAGNOSTIC(ImplAsNonFacetType, Error, "impl as non-facet type {0}",
                      InstIdAsType);
    context.emitter().Emit(impl.latest_decl_id(), ImplAsNonFacetType,
                           impl.constraint_id);
    return SemIR::ErrorInst::SingletonInstId;
  }
  const SemIR::FacetTypeInfo& facet_type_info =
      context.facet_types().Get(facet_type->facet_type_id);

  auto interface_type = facet_type_info.TryAsSingleInterface();
  if (!interface_type) {
    context.TODO(impl.latest_decl_id(), "impl as not 1 interface");
    return SemIR::ErrorInst::SingletonInstId;
  }
  const auto& interface =
      context.interfaces().Get(interface_type->interface_id);

  // TODO: This should be done as part of facet type resolution.
  if (!RequireDefinedType(context, facet_type_id,
                          context.insts().GetLocId(impl.latest_decl_id()), [&] {
                            CARBON_DIAGNOSTIC(
                                ImplOfUndefinedInterface, Error,
                                "implementation of undefined interface {0}",
                                SemIR::NameId);
                            return context.emitter().Build(
                                impl.latest_decl_id(), ImplOfUndefinedInterface,
                                interface.name_id);
                          })) {
    return SemIR::ErrorInst::SingletonInstId;
  }

  auto assoc_entities =
      context.inst_blocks().Get(interface.associated_entities_id);
  for (auto decl_id : assoc_entities) {
    LoadImportRef(context, decl_id);
  }

  llvm::SmallVector<SemIR::InstId> table(assoc_entities.size(),
                                         SemIR::InstId::None);
  auto table_id = context.inst_blocks().Add(table);
  return AddInst<SemIR::ImplWitness>(
      context, context.insts().GetLocId(impl.latest_decl_id()),
      {.type_id =
           GetSingletonType(context, SemIR::WitnessType::SingletonInstId),
       .elements_id = table_id,
       .specific_id = context.generics().GetSelfSpecific(impl.generic_id)});
}

// Returns `true` if the `FacetAccessWitness` of `witness_id` matches
// `interface`.
static auto WitnessAccessMatchesInterface(
    Context& context, SemIR::InstId witness_id,
    SemIR::FacetTypeInfo::ImplsConstraint interface) -> bool {
  auto access = context.insts().GetAs<SemIR::FacetAccessWitness>(witness_id);
  auto type_id = context.insts().Get(access.facet_value_inst_id).type_id();
  auto facet_type = context.types().GetAs<SemIR::FacetType>(type_id);
  const auto& facet_info = context.facet_types().Get(facet_type.facet_type_id);
  if (auto impls = facet_info.TryAsSingleInterface()) {
    return *impls == interface;
  }
  return false;
}

// TODO: Merge this function into `ImplWitnessForDeclaration`.
auto AddConstantsToImplWitnessFromConstraint(Context& context,
                                             const SemIR::Impl& impl,
                                             SemIR::InstId witness_id) -> void {
  CARBON_CHECK(!impl.has_definition_started());
  CARBON_CHECK(witness_id.has_value());
  if (witness_id == SemIR::ErrorInst::SingletonInstId) {
    return;
  }
  auto facet_type_id =
      context.types().GetTypeIdForTypeInstId(impl.constraint_id);
  CARBON_CHECK(facet_type_id != SemIR::ErrorInst::SingletonTypeId);
  auto facet_type = context.types().GetAs<SemIR::FacetType>(facet_type_id);
  const SemIR::FacetTypeInfo& facet_type_info =
      context.facet_types().Get(facet_type.facet_type_id);

  auto interface_type = facet_type_info.TryAsSingleInterface();
  CARBON_CHECK(interface_type.has_value());
  const auto& interface =
      context.interfaces().Get(interface_type->interface_id);

  auto witness = context.insts().GetAs<SemIR::ImplWitness>(witness_id);
  auto witness_block = context.inst_blocks().GetMutable(witness.elements_id);
  auto assoc_entities =
      context.inst_blocks().Get(interface.associated_entities_id);
  CARBON_CHECK(witness_block.size() == assoc_entities.size());

  // Scan through rewrites, produce map from element index to constant value.
  // TODO: Perhaps move this into facet type resolution?
  llvm::SmallVector<SemIR::ConstantId> rewrite_values(assoc_entities.size(),
                                                      SemIR::ConstantId::None);
  for (auto rewrite : facet_type_info.rewrite_constraints) {
    auto inst_id = context.constant_values().GetInstId(rewrite.lhs_const_id);
    auto access = context.insts().GetAs<SemIR::ImplWitnessAccess>(inst_id);
    if (!WitnessAccessMatchesInterface(context, access.witness_id,
                                       *interface_type)) {
      // Skip rewrite constraints that apply to associated constants of
      // a different interface than the one being implemented.
      continue;
    }
    CARBON_CHECK(access.index.index >= 0);
    CARBON_CHECK(access.index.index <
                 static_cast<int32_t>(rewrite_values.size()));
    auto& rewrite_value = rewrite_values[access.index.index];
    if (rewrite_value.has_value() &&
        rewrite_value != SemIR::ErrorInst::SingletonConstantId) {
      if (rewrite_value != rewrite.rhs_const_id &&
          rewrite.rhs_const_id != SemIR::ErrorInst::SingletonConstantId) {
        // TODO: Do at least this checking as part of facet type resolution
        // instead.

        // TODO: Figure out how to print the two different values
        // `rewrite_value` & `rewrite.rhs_const_id` in the diagnostic message.
        CARBON_DIAGNOSTIC(AssociatedConstantWithDifferentValues, Error,
                          "associated constant {0} given two different values",
                          SemIR::NameId);
        auto decl_id = context.constant_values().GetConstantInstId(
            assoc_entities[access.index.index]);
        CARBON_CHECK(decl_id.has_value(), "Non-constant associated entity");
        auto decl =
            context.insts().GetAs<SemIR::AssociatedConstantDecl>(decl_id);
        context.emitter().Emit(
            impl.constraint_id, AssociatedConstantWithDifferentValues,
            context.associated_constants().Get(decl.assoc_const_id).name_id);
      }
    } else {
      rewrite_value = rewrite.rhs_const_id;
    }
  }

  // For each non-function associated constant, set the witness entry.
  for (auto index : llvm::seq(assoc_entities.size())) {
    auto decl_id =
        context.constant_values().GetConstantInstId(assoc_entities[index]);
    CARBON_CHECK(decl_id.has_value(), "Non-constant associated entity");
    if (auto decl =
            context.insts().TryGetAs<SemIR::AssociatedConstantDecl>(decl_id)) {
      auto rewrite_value = rewrite_values[index];

      // If the associated constant has a symbolic type, convert the rewrite
      // value to that type now we know the value of `Self`.
      SemIR::TypeId assoc_const_type_id = decl->type_id;
      if (context.types().GetConstantId(assoc_const_type_id).is_symbolic()) {
        // Get the type of the associated constant in this interface with this
        // value for `Self`.
        assoc_const_type_id = GetTypeForSpecificAssociatedEntity(
            context, impl.constraint_id, interface_type->specific_id, decl_id,
            context.types().GetTypeIdForTypeInstId(impl.self_id), witness_id);

        // Perform the conversion of the value to the type. We skipped this when
        // forming the facet type because the type of the associated constant
        // was symbolic.
        auto converted_inst_id = ConvertToValueOfType(
            context, context.insts().GetLocId(impl.constraint_id),
            context.constant_values().GetInstId(rewrite_value),
            assoc_const_type_id);
        rewrite_value = context.constant_values().Get(converted_inst_id);

        // The result of conversion can be non-constant even if the original
        // value was constant.
        if (!rewrite_value.is_constant() &&
            rewrite_value != SemIR::ErrorInst::SingletonConstantId) {
          const auto& assoc_const =
              context.associated_constants().Get(decl->assoc_const_id);
          CARBON_DIAGNOSTIC(
              AssociatedConstantNotConstantAfterConversion, Error,
              "associated constant {0} given value that is not constant "
              "after conversion to {1}",
              SemIR::NameId, SemIR::TypeId);
          context.emitter().Emit(impl.constraint_id,
                                 AssociatedConstantNotConstantAfterConversion,
                                 assoc_const.name_id, assoc_const_type_id);
          rewrite_value = SemIR::ErrorInst::SingletonConstantId;
        }
      }

      if (rewrite_value.has_value()) {
        witness_block[index] =
            context.constant_values().GetInstId(rewrite_value);
      }
    }
  }
}

auto ImplWitnessStartDefinition(Context& context, SemIR::Impl& impl) -> void {
  CARBON_CHECK(impl.is_being_defined());
  CARBON_CHECK(impl.witness_id.has_value());
  if (impl.witness_id == SemIR::ErrorInst::SingletonInstId) {
    return;
  }

  auto facet_type_id =
      context.types().GetTypeIdForTypeInstId(impl.constraint_id);
  CARBON_CHECK(facet_type_id != SemIR::ErrorInst::SingletonTypeId);
  auto facet_type = context.types().GetAs<SemIR::FacetType>(facet_type_id);
  const SemIR::FacetTypeInfo& facet_type_info =
      context.facet_types().Get(facet_type.facet_type_id);

  auto interface_type = facet_type_info.TryAsSingleInterface();
  CARBON_CHECK(interface_type.has_value());
  const auto& interface =
      context.interfaces().Get(interface_type->interface_id);

  auto witness = context.insts().GetAs<SemIR::ImplWitness>(impl.witness_id);
  auto witness_block = context.inst_blocks().GetMutable(witness.elements_id);
  auto assoc_entities =
      context.inst_blocks().Get(interface.associated_entities_id);
  CARBON_CHECK(witness_block.size() == assoc_entities.size());

  // Check we have a value for all non-function associated constants in the
  // witness.
  for (auto index : llvm::seq(assoc_entities.size())) {
    auto decl_id = assoc_entities[index];
    decl_id = context.constant_values().GetConstantInstId(decl_id);
    CARBON_CHECK(decl_id.has_value(), "Non-constant associated entity");
    if (auto decl =
            context.insts().TryGetAs<SemIR::AssociatedConstantDecl>(decl_id)) {
      auto& witness_value = witness_block[index];
      if (!witness_value.has_value()) {
        CARBON_DIAGNOSTIC(ImplAssociatedConstantNeedsValue, Error,
                          "associated constant {0} not given a value in impl "
                          "of interface {1}",
                          SemIR::NameId, SemIR::NameId);
        CARBON_DIAGNOSTIC(AssociatedConstantHere, Note,
                          "associated constant declared here");
        context.emitter()
            .Build(impl.constraint_id, ImplAssociatedConstantNeedsValue,
                   context.associated_constants()
                       .Get(decl->assoc_const_id)
                       .name_id,
                   interface.name_id)
            .Note(assoc_entities[index], AssociatedConstantHere)
            .Emit();

        witness_value = SemIR::ErrorInst::SingletonInstId;
      }
    }
  }
}

// Adds functions to the witness that the specified impl implements the given
// interface.
auto FinishImplWitness(Context& context, SemIR::Impl& impl) -> void {
  CARBON_CHECK(impl.is_being_defined());
  CARBON_CHECK(impl.witness_id.has_value());
  if (impl.witness_id == SemIR::ErrorInst::SingletonInstId) {
    return;
  }

  auto facet_type_id =
      context.types().GetTypeIdForTypeInstId(impl.constraint_id);
  CARBON_CHECK(facet_type_id != SemIR::ErrorInst::SingletonTypeId);
  auto facet_type = context.types().GetAs<SemIR::FacetType>(facet_type_id);
  const SemIR::FacetTypeInfo& facet_type_info =
      context.facet_types().Get(facet_type.facet_type_id);

  auto interface_type = facet_type_info.TryAsSingleInterface();
  CARBON_CHECK(interface_type.has_value());
  const auto& interface =
      context.interfaces().Get(interface_type->interface_id);

  auto witness = context.insts().GetAs<SemIR::ImplWitness>(impl.witness_id);
  auto witness_block = context.inst_blocks().GetMutable(witness.elements_id);
  auto& impl_scope = context.name_scopes().Get(impl.scope_id);
  auto self_type_id = context.types().GetTypeIdForTypeInstId(impl.self_id);
  auto assoc_entities =
      context.inst_blocks().Get(interface.associated_entities_id);
  llvm::SmallVector<SemIR::InstId> used_decl_ids;

  for (auto index : llvm::seq(assoc_entities.size())) {
    auto decl_id = assoc_entities[index];
    decl_id =
        context.constant_values().GetInstId(SemIR::GetConstantValueInSpecific(
            context.sem_ir(), interface_type->specific_id, decl_id));
    CARBON_CHECK(decl_id.has_value(), "Non-constant associated entity");
    auto decl = context.insts().Get(decl_id);
    CARBON_KIND_SWITCH(decl) {
      case CARBON_KIND(SemIR::StructValue struct_value): {
        if (struct_value.type_id == SemIR::ErrorInst::SingletonTypeId) {
          witness_block[index] = SemIR::ErrorInst::SingletonInstId;
          break;
        }
        auto type_inst = context.types().GetAsInst(struct_value.type_id);
        auto fn_type = type_inst.TryAs<SemIR::FunctionType>();
        if (!fn_type) {
          CARBON_FATAL("Unexpected type: {0}", type_inst);
        }
        auto& fn = context.functions().Get(fn_type->function_id);
        auto lookup_result =
            LookupNameInExactScope(context, context.insts().GetLocId(decl_id),
                                   fn.name_id, impl.scope_id, impl_scope);
        if (lookup_result.is_found()) {
          used_decl_ids.push_back(lookup_result.target_inst_id());
          witness_block[index] = CheckAssociatedFunctionImplementation(
              context, *fn_type, lookup_result.target_inst_id(), self_type_id,
              impl.witness_id);
        } else {
          CARBON_DIAGNOSTIC(
              ImplMissingFunction, Error,
              "missing implementation of {0} in impl of interface {1}",
              SemIR::NameId, SemIR::NameId);
          auto builder =
              context.emitter().Build(impl.definition_id, ImplMissingFunction,
                                      fn.name_id, interface.name_id);
          NoteAssociatedFunction(context, builder, fn_type->function_id);
          builder.Emit();

          witness_block[index] = SemIR::ErrorInst::SingletonInstId;
        }
        break;
      }
      case SemIR::AssociatedConstantDecl::Kind: {
        // These are set to their final values already.
        break;
      }
      default:
        CARBON_CHECK(decl_id == SemIR::ErrorInst::SingletonInstId,
                     "Unexpected kind of associated entity {0}", decl);
        witness_block[index] = SemIR::ErrorInst::SingletonInstId;
        break;
    }
  }

  // TODO: Diagnose if any declarations in the impl are not in used_decl_ids.
}

auto FillImplWitnessWithErrors(Context& context, SemIR::Impl& impl) -> void {
  if (impl.witness_id.has_value() &&
      impl.witness_id != SemIR::ErrorInst::SingletonInstId) {
    auto witness = context.insts().GetAs<SemIR::ImplWitness>(impl.witness_id);
    auto witness_block = context.inst_blocks().GetMutable(witness.elements_id);
    for (auto& elem : witness_block) {
      if (!elem.has_value()) {
        elem = SemIR::ErrorInst::SingletonInstId;
      }
    }
  }
}

}  // namespace Carbon::Check
