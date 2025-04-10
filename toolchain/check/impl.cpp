// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/impl.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/facet_type.h"
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
static auto NoteAssociatedFunction(Context& context, DiagnosticBuilder& builder,
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

  auto impl_enclosing_specific_id =
      context.types()
          .GetAs<SemIR::FunctionType>(impl_function_decl->type_id)
          .specific_id;

  // Map from the specific for the function type to the specific for the
  // function signature. The function signature may have additional generic
  // parameters.
  auto interface_function_specific_id =
      GetSelfSpecificForInterfaceMemberWithSelfType(
          context, impl_decl_id, interface_function_type.specific_id,
          context.functions()
              .Get(interface_function_type.function_id)
              .generic_id,
          impl_enclosing_specific_id, self_type_id, witness_inst_id);

  if (!CheckFunctionTypeMatches(
          context, context.functions().Get(impl_function_decl->function_id),
          context.functions().Get(interface_function_type.function_id),
          interface_function_specific_id, /*check_syntax=*/false,
          /*check_self=*/true)) {
    return SemIR::ErrorInst::SingletonInstId;
  }
  return impl_decl_id;
}

// Builds an initial witness from the rewrites in the facet type, if any.
auto ImplWitnessForDeclaration(Context& context, const SemIR::Impl& impl,
                               bool has_definition) -> SemIR::InstId {
  CARBON_CHECK(!impl.has_definition_started());

  auto self_type_id = context.types().GetTypeIdForTypeInstId(impl.self_id);
  if (self_type_id == SemIR::ErrorInst::SingletonTypeId) {
    // When 'impl as' is invalid, the self type is an error.
    return SemIR::ErrorInst::SingletonInstId;
  }

  return InitialFacetTypeImplWitness(
      context, context.insts().GetLocId(impl.latest_decl_id()),
      impl.constraint_id, impl.self_id, impl.interface,
      context.generics().GetSelfSpecific(impl.generic_id), has_definition);
}

auto ImplWitnessStartDefinition(Context& context, SemIR::Impl& impl) -> void {
  CARBON_CHECK(impl.is_being_defined());
  CARBON_CHECK(impl.witness_id.has_value());
  if (impl.witness_id == SemIR::ErrorInst::SingletonInstId) {
    return;
  }
  auto witness = context.insts().GetAs<SemIR::ImplWitness>(impl.witness_id);
  auto witness_block = context.inst_blocks().GetMutable(witness.elements_id);
  // `witness.elements_id` will be `SemIR::InstBlockId::Empty` when the
  // definition is the first declaration and the interface has no members. The
  // other case where `witness_block` will be empty is when we are using a
  // placeholder witness. This happens when there is a forward declaration of
  // the impl and the facet type has no rewrite constraints and so it wasn't
  // required to be complete.
  if (witness.elements_id != SemIR::InstBlockId::Empty &&
      witness_block.empty()) {
    if (!RequireCompleteFacetTypeForImplDefinition(
            context, impl.latest_decl_id(), impl.constraint_id)) {
      return;
    }

    AllocateFacetTypeImplWitness(context, impl.interface.interface_id,
                                 witness.elements_id);
    witness_block = context.inst_blocks().GetMutable(witness.elements_id);
  }
  const auto& interface = context.interfaces().Get(impl.interface.interface_id);
  auto assoc_entities =
      context.inst_blocks().Get(interface.associated_entities_id);
  CARBON_CHECK(witness_block.size() == assoc_entities.size());

  // Check we have a value for all non-function associated constants in the
  // witness.
  for (auto [assoc_entity, witness_value] :
       llvm::zip(assoc_entities, witness_block)) {
    auto decl_id = context.constant_values().GetConstantInstId(assoc_entity);
    CARBON_CHECK(decl_id.has_value(), "Non-constant associated entity");
    if (auto decl =
            context.insts().TryGetAs<SemIR::AssociatedConstantDecl>(decl_id)) {
      if (witness_value ==
          SemIR::ImplWitnessTablePlaceholder::SingletonInstId) {
        CARBON_DIAGNOSTIC(ImplAssociatedConstantNeedsValue, Error,
                          "associated constant {0} not given a value in impl "
                          "of interface {1}",
                          SemIR::NameId, SemIR::NameId);
        CARBON_DIAGNOSTIC(AssociatedConstantHere, Note,
                          "associated constant declared here");
        context.emitter()
            .Build(impl.definition_id, ImplAssociatedConstantNeedsValue,
                   context.associated_constants()
                       .Get(decl->assoc_const_id)
                       .name_id,
                   interface.name_id)
            .Note(assoc_entity, AssociatedConstantHere)
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
  auto witness = context.insts().GetAs<SemIR::ImplWitness>(impl.witness_id);
  auto witness_block = context.inst_blocks().GetMutable(witness.elements_id);
  auto& impl_scope = context.name_scopes().Get(impl.scope_id);
  auto self_type_id = context.types().GetTypeIdForTypeInstId(impl.self_id);
  const auto& interface = context.interfaces().Get(impl.interface.interface_id);
  auto assoc_entities =
      context.inst_blocks().Get(interface.associated_entities_id);
  llvm::SmallVector<SemIR::InstId> used_decl_ids;

  for (auto [assoc_entity, witness_value] :
       llvm::zip(assoc_entities, witness_block)) {
    auto decl_id =
        context.constant_values().GetInstId(SemIR::GetConstantValueInSpecific(
            context.sem_ir(), impl.interface.specific_id, assoc_entity));
    CARBON_CHECK(decl_id.has_value(), "Non-constant associated entity");
    auto decl = context.insts().Get(decl_id);
    CARBON_KIND_SWITCH(decl) {
      case CARBON_KIND(SemIR::StructValue struct_value): {
        if (struct_value.type_id == SemIR::ErrorInst::SingletonTypeId) {
          witness_value = SemIR::ErrorInst::SingletonInstId;
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
          witness_value = CheckAssociatedFunctionImplementation(
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

          witness_value = SemIR::ErrorInst::SingletonInstId;
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
        witness_value = SemIR::ErrorInst::SingletonInstId;
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
      if (elem == SemIR::ImplWitnessTablePlaceholder::SingletonInstId) {
        elem = SemIR::ErrorInst::SingletonInstId;
      }
    }
  }
}

}  // namespace Carbon::Check
