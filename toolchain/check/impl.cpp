// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/impl.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/function.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/import_ref.h"
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

// Adds the location of the previous declaration to a diagnostic.
static auto NotePreviousDecl(Context& context,
                             Context::DiagnosticBuilder& builder,
                             SemIR::ImplId impl_id) -> void {
  CARBON_DIAGNOSTIC(ImplPreviousDeclHere, Note,
                    "impl previously declared here");
  const auto& impl = context.impls().Get(impl_id);
  builder.Note(impl.latest_decl_id(), ImplPreviousDeclHere);
}

// Gets the self specific of a generic declaration that is an interface member,
// given a specific for an enclosing generic, plus a type to use as `Self`.
static auto GetSelfSpecificForInterfaceMemberWithSelfType(
    Context& context, SemIRLoc loc, SemIR::SpecificId enclosing_specific_id,
    SemIR::GenericId generic_id, SemIR::TypeId self_type_id,
    SemIR::InstId witness_inst_id) -> SemIR::SpecificId {
  const auto& generic = context.generics().Get(generic_id);
  auto bindings = context.inst_blocks().Get(generic.bindings_id);

  llvm::SmallVector<SemIR::InstId> arg_ids;
  arg_ids.reserve(bindings.size());

  // Start with the enclosing arguments.
  if (enclosing_specific_id.is_valid()) {
    auto enclosing_specific_args_id =
        context.specifics().Get(enclosing_specific_id).args_id;
    auto enclosing_specific_args =
        context.inst_blocks().Get(enclosing_specific_args_id);
    arg_ids.assign(enclosing_specific_args.begin(),
                   enclosing_specific_args.end());
  }

  // Add the `Self` argument. First find the `Self` binding.
  auto self_binding =
      context.insts().GetAs<SemIR::BindSymbolicName>(bindings[arg_ids.size()]);
  CARBON_CHECK(
      context.entity_names().Get(self_binding.entity_name_id).name_id ==
          SemIR::NameId::SelfType,
      "Expected a Self binding, found {0}", self_binding);
  // Create a facet value to be the value of `Self` in the interface.
  // This facet value consists of the type `self_type_id` and a witness that the
  // type implements `self_binding.type_id`. Note that the witness is incomplete
  // since we haven't finished defining the implementation here.
  auto type_inst_id = context.types().GetInstId(self_type_id);
  auto facet_value_const_id =
      TryEvalInst(context, SemIR::InstId::Invalid,
                  SemIR::FacetValue{.type_id = self_binding.type_id,
                                    .type_inst_id = type_inst_id,
                                    .witness_inst_id = witness_inst_id});
  arg_ids.push_back(context.constant_values().GetInstId(facet_value_const_id));

  // Take any trailing argument values from the self specific.
  // TODO: If these refer to outer arguments, for example in their types, we may
  // need to perform extra substitutions here.
  auto self_specific_args = context.inst_blocks().Get(
      context.specifics().Get(generic.self_specific_id).args_id);
  for (auto arg_id : self_specific_args.drop_front(arg_ids.size())) {
    arg_ids.push_back(context.constant_values().GetConstantInstId(arg_id));
  }

  auto args_id = context.inst_blocks().AddCanonical(arg_ids);
  return MakeSpecific(context, loc, generic_id, args_id);
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

  // TODO: This should be a semantic check rather than a syntactic one. The
  // functions should be allowed to have different signatures as long as we can
  // synthesize a suitable thunk.
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
auto ImplWitnessForDeclaration(Context& context, const SemIR::Impl& impl)
    -> SemIR::InstId {
  CARBON_CHECK(!impl.has_definition_started());

  auto facet_type_id = context.GetTypeIdForTypeInst(impl.constraint_id);
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
  if (!context.RequireDefinedType(
          facet_type_id, context.insts().GetLocId(impl.latest_decl_id()), [&] {
            CARBON_DIAGNOSTIC(ImplOfUndefinedInterface, Error,
                              "implementation of undefined interface {0}",
                              SemIR::NameId);
            return context.emitter().Build(impl.latest_decl_id(),
                                           ImplOfUndefinedInterface,
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
                                         SemIR::InstId::Invalid);
  auto table_id = context.inst_blocks().Add(table);
  return context.AddInst<SemIR::ImplWitness>(
      context.insts().GetLocId(impl.latest_decl_id()),
      {.type_id = context.GetSingletonType(SemIR::WitnessType::SingletonInstId),
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

auto AddConstantsToImplWitnessFromConstraint(Context& context,
                                             const SemIR::Impl& impl,
                                             SemIR::InstId witness_id,
                                             SemIR::ImplId prev_decl_id)
    -> void {
  CARBON_CHECK(!impl.has_definition_started());
  CARBON_CHECK(witness_id.is_valid());
  if (witness_id == SemIR::ErrorInst::SingletonInstId) {
    return;
  }
  auto facet_type_id = context.GetTypeIdForTypeInst(impl.constraint_id);
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
  llvm::SmallVector<SemIR::ConstantId> rewrite_values(
      assoc_entities.size(), SemIR::ConstantId::Invalid);
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
    if (rewrite_value.is_valid() &&
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
        auto decl_id = assoc_entities[access.index.index];
        decl_id = context.constant_values().GetInstId(
            SemIR::GetConstantValueInSpecific(
                context.sem_ir(), interface_type->specific_id, decl_id));
        CARBON_CHECK(decl_id.is_valid(), "Non-constant associated entity");
        auto decl =
            context.insts().GetAs<SemIR::AssociatedConstantDecl>(decl_id);
        context.emitter().Emit(impl.constraint_id,
                               AssociatedConstantWithDifferentValues,
                               decl.name_id);
      }
    } else {
      rewrite_value = rewrite.rhs_const_id;
    }
  }

  // For each non-function associated constant, update witness entry.
  for (auto index : llvm::seq(assoc_entities.size())) {
    auto decl_id = assoc_entities[index];
    decl_id =
        context.constant_values().GetInstId(SemIR::GetConstantValueInSpecific(
            context.sem_ir(), interface_type->specific_id, decl_id));
    CARBON_CHECK(decl_id.is_valid(), "Non-constant associated entity");
    if (auto decl =
            context.insts().TryGetAs<SemIR::AssociatedConstantDecl>(decl_id)) {
      auto& witness_value = witness_block[index];
      auto rewrite_value = rewrite_values[index];
      if (witness_value.is_valid() &&
          witness_value != SemIR::ErrorInst::SingletonInstId) {
        // TODO: Support just using the witness values if the redeclaration uses
        // `where _`, per proposal #1084.
        if (!rewrite_value.is_valid()) {
          CARBON_DIAGNOSTIC(AssociatedConstantMissingInRedecl, Error,
                            "associated constant {0} given value in "
                            "declaration but not redeclaration",
                            SemIR::NameId);
          auto builder = context.emitter().Build(
              impl.latest_decl_id(), AssociatedConstantMissingInRedecl,
              decl->name_id);
          NotePreviousDecl(context, builder, prev_decl_id);
          builder.Emit();
          continue;
        }
        auto witness_const_id = context.constant_values().Get(witness_value);
        if (witness_const_id != rewrite_value &&
            rewrite_value != SemIR::ErrorInst::SingletonConstantId) {
          // TODO: Figure out how to print the two different values
          CARBON_DIAGNOSTIC(
              AssociatedConstantDifferentInRedecl, Error,
              "redeclaration with different value for associated constant {0}",
              SemIR::NameId);
          auto builder = context.emitter().Build(
              impl.latest_decl_id(), AssociatedConstantDifferentInRedecl,
              decl->name_id);
          NotePreviousDecl(context, builder, prev_decl_id);
          builder.Emit();
          continue;
        }
      } else if (rewrite_value.is_valid()) {
        witness_value = context.constant_values().GetInstId(rewrite_value);
      }
    }
  }
}

auto ImplWitnessStartDefinition(Context& context, SemIR::Impl& impl) -> void {
  CARBON_CHECK(impl.is_being_defined());
  CARBON_CHECK(impl.witness_id.is_valid());
  if (impl.witness_id == SemIR::ErrorInst::SingletonInstId) {
    return;
  }

  auto facet_type_id = context.GetTypeIdForTypeInst(impl.constraint_id);
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
    decl_id =
        context.constant_values().GetInstId(SemIR::GetConstantValueInSpecific(
            context.sem_ir(), interface_type->specific_id, decl_id));
    CARBON_CHECK(decl_id.is_valid(), "Non-constant associated entity");
    if (auto decl =
            context.insts().TryGetAs<SemIR::AssociatedConstantDecl>(decl_id)) {
      auto& witness_value = witness_block[index];
      if (!witness_value.is_valid()) {
        CARBON_DIAGNOSTIC(ImplAssociatedConstantNeedsValue, Error,
                          "associated constant {0} not given a value in impl "
                          "of interface {1}",
                          SemIR::NameId, SemIR::NameId);
        CARBON_DIAGNOSTIC(AssociatedConstantHere, Note,
                          "associated constant {0} declared here",
                          SemIR::NameId);
        context.emitter()
            .Build(impl.constraint_id, ImplAssociatedConstantNeedsValue,
                   decl->name_id, interface.name_id)
            .Note(assoc_entities[index], AssociatedConstantHere, decl->name_id)
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
  CARBON_CHECK(impl.witness_id.is_valid());
  if (impl.witness_id == SemIR::ErrorInst::SingletonInstId) {
    return;
  }

  auto facet_type_id = context.GetTypeIdForTypeInst(impl.constraint_id);
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
  auto self_type_id = context.GetTypeIdForTypeInst(impl.self_id);
  auto assoc_entities =
      context.inst_blocks().Get(interface.associated_entities_id);
  llvm::SmallVector<SemIR::InstId> used_decl_ids;

  for (auto index : llvm::seq(assoc_entities.size())) {
    auto decl_id = assoc_entities[index];
    decl_id =
        context.constant_values().GetInstId(SemIR::GetConstantValueInSpecific(
            context.sem_ir(), interface_type->specific_id, decl_id));
    CARBON_CHECK(decl_id.is_valid(), "Non-constant associated entity");
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
        auto [impl_decl_id, _, is_poisoned] = context.LookupNameInExactScope(
            decl_id, fn.name_id, impl.scope_id, impl_scope);
        if (impl_decl_id.is_valid()) {
          used_decl_ids.push_back(impl_decl_id);
          witness_block[index] = CheckAssociatedFunctionImplementation(
              context, *fn_type, impl_decl_id, self_type_id, impl.witness_id);
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
  if (impl.witness_id.is_valid() &&
      impl.witness_id != SemIR::ErrorInst::SingletonInstId) {
    auto witness = context.insts().GetAs<SemIR::ImplWitness>(impl.witness_id);
    auto witness_block = context.inst_blocks().GetMutable(witness.elements_id);
    for (auto& elem : witness_block) {
      if (!elem.is_valid()) {
        elem = SemIR::ErrorInst::SingletonInstId;
      }
    }
  }
}

}  // namespace Carbon::Check
