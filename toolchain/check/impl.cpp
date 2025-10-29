// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/impl.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/deduce.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/facet_type.h"
#include "toolchain/check/function.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/interface.h"
#include "toolchain/check/merge.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/thunk.h"
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

    return SemIR::ErrorInst::InstId;
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
          context, SemIR::LocId(impl_decl_id),
          interface_function_type.specific_id,
          context.functions()
              .Get(interface_function_type.function_id)
              .generic_id,
          impl_enclosing_specific_id, self_type_id, witness_inst_id);

  return BuildThunk(context, interface_function_type.function_id,
                    interface_function_specific_id, impl_decl_id);
}

// Builds an initial witness from the rewrites in the facet type, if any.
auto ImplWitnessForDeclaration(Context& context, const SemIR::Impl& impl,
                               bool has_definition) -> SemIR::InstId {
  CARBON_CHECK(!impl.has_definition_started());

  auto self_type_id = context.types().GetTypeIdForTypeInstId(impl.self_id);
  if (self_type_id == SemIR::ErrorInst::TypeId) {
    // When 'impl as' is invalid, the self type is an error.
    return SemIR::ErrorInst::InstId;
  }

  return InitialFacetTypeImplWitness(
      context, SemIR::LocId(impl.latest_decl_id()), impl.constraint_id,
      impl.self_id, impl.interface,
      context.generics().GetSelfSpecific(impl.generic_id), has_definition);
}

auto ImplWitnessStartDefinition(Context& context, SemIR::Impl& impl) -> void {
  CARBON_CHECK(impl.is_being_defined());
  CARBON_CHECK(impl.witness_id.has_value());
  if (impl.witness_id == SemIR::ErrorInst::InstId) {
    return;
  }
  auto witness = context.insts().GetAs<SemIR::ImplWitness>(impl.witness_id);
  auto witness_table =
      context.insts().GetAs<SemIR::ImplWitnessTable>(witness.witness_table_id);
  auto witness_block =
      context.inst_blocks().GetMutable(witness_table.elements_id);
  // `witness_table.elements_id` will be `SemIR::InstBlockId::Empty` when the
  // definition is the first declaration and the interface has no members. The
  // other case where `witness_block` will be empty is when we are using a
  // placeholder witness. This happens when there is a forward declaration of
  // the impl and the facet type has no rewrite constraints and so it wasn't
  // required to be complete.
  if (witness_table.elements_id != SemIR::InstBlockId::Empty &&
      witness_block.empty()) {
    if (!RequireCompleteFacetTypeForImplDefinition(
            context, SemIR::LocId(impl.latest_decl_id()), impl.constraint_id)) {
      return;
    }

    AllocateFacetTypeImplWitness(context, impl.interface.interface_id,
                                 witness_table.elements_id);
    witness_block = context.inst_blocks().GetMutable(witness_table.elements_id);
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
      if (witness_value == SemIR::InstId::ImplWitnessTablePlaceholder) {
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

        witness_value = SemIR::ErrorInst::InstId;
      }
    }
  }
}

// Adds functions to the witness that the specified impl implements the given
// interface.
auto FinishImplWitness(Context& context, SemIR::ImplId impl_id) -> void {
  const auto& impl = context.impls().Get(impl_id);

  CARBON_CHECK(impl.is_being_defined());
  CARBON_CHECK(impl.witness_id.has_value());
  if (impl.witness_id == SemIR::ErrorInst::InstId) {
    return;
  }
  auto witness = context.insts().GetAs<SemIR::ImplWitness>(impl.witness_id);
  auto witness_table =
      context.insts().GetAs<SemIR::ImplWitnessTable>(witness.witness_table_id);
  auto witness_block =
      context.inst_blocks().GetMutable(witness_table.elements_id);
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
        if (struct_value.type_id == SemIR::ErrorInst::TypeId) {
          witness_value = SemIR::ErrorInst::InstId;
          break;
        }
        auto type_inst = context.types().GetAsInst(struct_value.type_id);
        auto fn_type = type_inst.TryAs<SemIR::FunctionType>();
        if (!fn_type) {
          CARBON_FATAL("Unexpected type: {0}", type_inst);
        }
        auto& fn = context.functions().Get(fn_type->function_id);
        auto lookup_result =
            LookupNameInExactScope(context, SemIR::LocId(decl_id), fn.name_id,
                                   impl.scope_id, impl_scope);
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

          witness_value = SemIR::ErrorInst::InstId;
        }
        break;
      }
      case SemIR::AssociatedConstantDecl::Kind: {
        // These are set to their final values already.
        break;
      }
      default:
        CARBON_CHECK(decl_id == SemIR::ErrorInst::InstId,
                     "Unexpected kind of associated entity {0}", decl);
        witness_value = SemIR::ErrorInst::InstId;
        break;
    }
  }

  // TODO: Diagnose if any declarations in the impl are not in used_decl_ids.
}

auto FillImplWitnessWithErrors(Context& context, SemIR::Impl& impl) -> void {
  if (impl.witness_id == SemIR::ErrorInst::InstId) {
    return;
  }
  auto witness = context.insts().GetAs<SemIR::ImplWitness>(impl.witness_id);
  auto witness_table =
      context.insts().GetAs<SemIR::ImplWitnessTable>(witness.witness_table_id);
  auto witness_block =
      context.inst_blocks().GetMutable(witness_table.elements_id);
  for (auto& elem : witness_block) {
    if (elem == SemIR::InstId::ImplWitnessTablePlaceholder) {
      elem = SemIR::ErrorInst::InstId;
    }
  }
  impl.witness_id = SemIR::ErrorInst::InstId;
}

auto AssignImplIdInWitness(Context& context, SemIR::ImplId impl_id,
                           SemIR::InstId witness_id) -> void {
  if (witness_id == SemIR::ErrorInst::InstId) {
    return;
  }
  auto witness = context.insts().GetAs<SemIR::ImplWitness>(witness_id);
  auto witness_table =
      context.insts().GetAs<SemIR::ImplWitnessTable>(witness.witness_table_id);
  witness_table.impl_id = impl_id;
  // Note: The `ImplWitnessTable` instruction is `Unique`, so while this marks
  // the instruction as being a dependent instruction of a generic impl, it will
  // not be substituted into the eval block.
  ReplaceInstBeforeConstantUse(context, witness.witness_table_id,
                               witness_table);
}

auto IsImplEffectivelyFinal(Context& context, const SemIR::Impl& impl) -> bool {
  return impl.is_final ||
         (context.constant_values().Get(impl.self_id).is_concrete() &&
          context.constant_values().Get(impl.constraint_id).is_concrete());
}

auto CheckConstraintIsInterface(Context& context, SemIR::InstId impl_decl_id,
                                SemIR::TypeInstId constraint_id)
    -> SemIR::SpecificInterface {
  auto facet_type_id = context.types().GetTypeIdForTypeInstId(constraint_id);
  if (facet_type_id == SemIR::ErrorInst::TypeId) {
    return SemIR::SpecificInterface::None;
  }
  auto facet_type = context.types().TryGetAs<SemIR::FacetType>(facet_type_id);
  if (!facet_type) {
    CARBON_DIAGNOSTIC(ImplAsNonFacetType, Error, "impl as non-facet type {0}",
                      InstIdAsType);
    context.emitter().Emit(impl_decl_id, ImplAsNonFacetType, constraint_id);
    return SemIR::SpecificInterface::None;
  }

  auto identified_id = RequireIdentifiedFacetType(context, *facet_type);
  const auto& identified = context.identified_facet_types().Get(identified_id);
  if (!identified.is_valid_impl_as_target()) {
    CARBON_DIAGNOSTIC(ImplOfNotOneInterface, Error,
                      "impl as {0} interfaces, expected 1", int);
    context.emitter().Emit(impl_decl_id, ImplOfNotOneInterface,
                           identified.num_interfaces_to_impl());
    return SemIR::SpecificInterface::None;
  }
  return identified.impl_as_target_interface();
}

// Returns true if impl redeclaration parameters match.
static auto CheckImplRedeclParamsMatch(Context& context, SemIR::Impl& new_impl,
                                       SemIR::ImplId prev_impl_id) -> bool {
  auto& prev_impl = context.impls().Get(prev_impl_id);

  // If the parameters aren't the same, then this is not a redeclaration of this
  // `impl`. Keep looking for a prior declaration without issuing a diagnostic.
  if (!CheckRedeclParamsMatch(context, DeclParams(new_impl),
                              DeclParams(prev_impl), SemIR::SpecificId::None,
                              /*diagnose=*/false, /*check_syntax=*/true,
                              /*check_self=*/true)) {
    // NOLINTNEXTLINE(readability-simplify-boolean-expr)
    return false;
  }
  return true;
}

// Returns whether an impl can be redeclared. For example, defined impls
// cannot be redeclared.
static auto IsValidImplRedecl(Context& context, SemIR::Impl& new_impl,
                              SemIR::ImplId prev_impl_id) -> bool {
  auto& prev_impl = context.impls().Get(prev_impl_id);

  // TODO: Following #3763, disallow redeclarations in different scopes.

  // Following #4672, disallowing defining non-extern declarations in another
  // file.
  if (auto import_ref =
          context.insts().TryGetAs<SemIR::AnyImportRef>(prev_impl.self_id)) {
    // TODO: Handle extern.
    CARBON_DIAGNOSTIC(RedeclImportedImpl, Error,
                      "redeclaration of imported impl");
    // TODO: Note imported declaration
    context.emitter().Emit(new_impl.latest_decl_id(), RedeclImportedImpl);
    return false;
  }

  if (prev_impl.has_definition_started()) {
    // Impls aren't merged in order to avoid generic region lookup into a
    // mismatching table.
    CARBON_DIAGNOSTIC(ImplRedefinition, Error,
                      "redefinition of `impl {0} as {1}`", InstIdAsRawType,
                      InstIdAsRawType);
    CARBON_DIAGNOSTIC(ImplPreviousDefinition, Note,
                      "previous definition was here");
    context.emitter()
        .Build(new_impl.latest_decl_id(), ImplRedefinition, new_impl.self_id,
               new_impl.constraint_id)
        .Note(prev_impl.definition_id, ImplPreviousDefinition)
        .Emit();
    return false;
  }

  // TODO: Only allow redeclaration in a match_first/impl_priority block.

  return true;
}

static auto DiagnoseExtendImplOutsideClass(Context& context,
                                           SemIR::LocId loc_id) -> void {
  CARBON_DIAGNOSTIC(ExtendImplOutsideClass, Error,
                    "`extend impl` can only be used in a class");
  context.emitter().Emit(loc_id, ExtendImplOutsideClass);
}

// If the specified name scope corresponds to a class, returns the corresponding
// class declaration.
// TODO: Should this be somewhere more central?
static auto TryAsClassScope(Context& context, SemIR::NameScopeId scope_id)
    -> std::optional<SemIR::ClassDecl> {
  if (!scope_id.has_value()) {
    return std::nullopt;
  }
  auto& scope = context.name_scopes().Get(scope_id);
  if (!scope.inst_id().has_value()) {
    return std::nullopt;
  }
  return context.insts().TryGetAs<SemIR::ClassDecl>(scope.inst_id());
}

auto GetImplDefaultSelfType(Context& context) -> SemIR::TypeId {
  auto parent_scope_id = context.decl_name_stack().PeekParentScopeId();

  if (auto class_decl = TryAsClassScope(context, parent_scope_id)) {
    return context.classes().Get(class_decl->class_id).self_type_id;
  }

  // TODO: This is also valid in a mixin.

  return SemIR::TypeId::None;
}

// Process an `extend impl` declaration by extending the impl scope with the
// `impl`'s scope.
static auto ExtendImpl(Context& context, Parse::NodeId extend_node,
                       SemIR::LocId loc_id, SemIR::ImplId impl_id,
                       Parse::NodeId self_type_node_id,
                       SemIR::TypeId self_type_id,
                       SemIR::LocId implicit_params_loc_id,
                       SemIR::TypeInstId constraint_type_inst_id,
                       SemIR::TypeId constraint_type_id) -> bool {
  auto parent_scope_id = context.decl_name_stack().PeekParentScopeId();
  if (!parent_scope_id.has_value()) {
    DiagnoseExtendImplOutsideClass(context, loc_id);
    return false;
  }
  // TODO: This is also valid in a mixin.
  if (!TryAsClassScope(context, parent_scope_id)) {
    DiagnoseExtendImplOutsideClass(context, loc_id);
    return false;
  }

  auto& parent_scope = context.name_scopes().Get(parent_scope_id);

  if (implicit_params_loc_id.has_value()) {
    CARBON_DIAGNOSTIC(ExtendImplForall, Error,
                      "cannot `extend` a parameterized `impl`");
    context.emitter().Emit(extend_node, ExtendImplForall);
    parent_scope.set_has_error();
    return false;
  }

  const auto& impl = context.impls().Get(impl_id);

  if (context.parse_tree().node_kind(self_type_node_id) ==
      Parse::NodeKind::ImplTypeAs) {
    CARBON_DIAGNOSTIC(ExtendImplSelfAs, Error,
                      "cannot `extend` an `impl` with an explicit self type");
    auto diag = context.emitter().Build(extend_node, ExtendImplSelfAs);

    // If the explicit self type is not the default, just bail out.
    if (self_type_id != GetImplDefaultSelfType(context)) {
      diag.Emit();
      parent_scope.set_has_error();
      return false;
    }

    // The explicit self type is the same as the default self type, so suggest
    // removing it and recover as if it were not present.
    if (auto self_as =
            context.parse_tree_and_subtrees().ExtractAs<Parse::ImplTypeAs>(
                self_type_node_id)) {
      CARBON_DIAGNOSTIC(ExtendImplSelfAsDefault, Note,
                        "remove the explicit `Self` type here");
      diag.Note(self_as->type_expr, ExtendImplSelfAsDefault);
    }
    diag.Emit();
  }

  if (impl.witness_id == SemIR::ErrorInst::InstId) {
    parent_scope.set_has_error();
  } else {
    bool is_complete = RequireCompleteType(
        context, constraint_type_id, SemIR::LocId(constraint_type_inst_id),
        [&] {
          CARBON_DIAGNOSTIC(ExtendImplAsIncomplete, Error,
                            "`extend impl as` incomplete facet type {0}",
                            InstIdAsType);
          return context.emitter().Build(impl.latest_decl_id(),
                                         ExtendImplAsIncomplete,
                                         constraint_type_inst_id);
        });
    if (!is_complete) {
      parent_scope.set_has_error();
      return false;
    }
  }

  parent_scope.AddExtendedScope(constraint_type_inst_id);
  return true;
}

// Diagnoses when an impl has an unused binding.
static auto DiagnoseUnusedGenericBinding(Context& context, SemIR::LocId loc_id,
                                         SemIR::LocId implicit_params_loc_id,
                                         SemIR::ImplId impl_id) -> void {
  auto& impl = context.impls().Get(impl_id);
  if (!impl.generic_id.has_value() ||
      impl.witness_id == SemIR::ErrorInst::InstId) {
    return;
  }

  auto deduced_specific_id = DeduceImplArguments(
      context, loc_id, impl, context.constant_values().Get(impl.self_id),
      impl.interface.specific_id);
  if (deduced_specific_id.has_value()) {
    // Deduction succeeded, all bindings were used.
    return;
  }

  CARBON_DIAGNOSTIC(ImplUnusedBinding, Error,
                    "`impl` with unused generic binding");
  // TODO: This location may be incorrect, the binding may be inherited
  // from an outer declaration. It would be nice to get the particular
  // binding that was undeducible back from DeduceImplArguments here and
  // use that.
  auto diag_loc_id =
      implicit_params_loc_id.has_value() ? implicit_params_loc_id : loc_id;
  context.emitter().Emit(diag_loc_id, ImplUnusedBinding);
  // Don't try to match the impl at all, save us work and possible future
  // diagnostics.
  FillImplWitnessWithErrors(context, context.impls().Get(impl_id));
}

auto StartImplDecl(Context& context, SemIR::LocId loc_id,
                   SemIR::LocId implicit_params_loc_id, SemIR::Impl impl,
                   bool is_definition,
                   std::optional<ExtendImplDecl> extend_impl)
    -> std::pair<SemIR::ImplId, SemIR::InstId> {
  auto impl_id = SemIR::ImplId::None;

  // Add the impl declaration.
  auto lookup_bucket_ref = context.impls().GetOrAddLookupBucket(impl);
  // TODO: Detect two impl declarations with the same self type and interface,
  // and issue an error if they don't match.
  for (auto prev_impl_id : lookup_bucket_ref) {
    if (CheckImplRedeclParamsMatch(context, impl, prev_impl_id)) {
      if (IsValidImplRedecl(context, impl, prev_impl_id)) {
        impl_id = prev_impl_id;
      } else {
        // IsValidImplRedecl() has issued a diagnostic, avoid generating more
        // diagnostics for this declaration.
        impl.witness_id = SemIR::ErrorInst::InstId;
      }
      break;
    }
  }

  // Create a new impl if this isn't a valid redeclaration.
  if (!impl_id.has_value()) {
    impl.generic_id = BuildGeneric(context, impl.latest_decl_id());
    if (impl.witness_id != SemIR::ErrorInst::InstId) {
      if (impl.interface.interface_id.has_value()) {
        impl.witness_id =
            ImplWitnessForDeclaration(context, impl, is_definition);
      } else {
        impl.witness_id = SemIR::ErrorInst::InstId;
        // TODO: We might also want to mark that the name scope for the impl has
        // an error -- at least once we start making name lookups within the
        // impl also look into the facet (eg, so you can name associated
        // constants from within the impl).
      }
    }
    FinishGenericDecl(context, SemIR::LocId(impl.latest_decl_id()),
                      impl.generic_id);
    // From here on, use the `Impl` from the `ImplStore` instead of `impl`
    // in order to make and see any changes to the `Impl`.
    impl_id = context.impls().Add(impl);
    lookup_bucket_ref.push_back(impl_id);

    AssignImplIdInWitness(context, impl_id, impl.witness_id);

    // Looking to see if there are any generic bindings on the `impl`
    // declaration that are not deducible. If so, and the `impl` does not
    // actually use all its generic bindings, and will never be matched. This
    // should be diagnossed to the user.
    bool has_error_in_implicit_pattern = false;
    if (impl.implicit_param_patterns_id.has_value()) {
      for (auto inst_id :
           context.inst_blocks().Get(impl.implicit_param_patterns_id)) {
        if (inst_id == SemIR::ErrorInst::InstId) {
          has_error_in_implicit_pattern = true;
          break;
        }
      }
    }

    if (!has_error_in_implicit_pattern) {
      DiagnoseUnusedGenericBinding(context, loc_id, implicit_params_loc_id,
                                   impl_id);
    }
  } else {
    auto& stored_impl = context.impls().Get(impl_id);
    FinishGenericRedecl(context, stored_impl.generic_id);
  }

  // Write the impl ID into the ImplDecl.
  auto impl_decl =
      context.insts().GetAs<SemIR::ImplDecl>(impl.first_owning_decl_id);
  CARBON_CHECK(!impl_decl.impl_id.has_value());
  impl_decl.impl_id = impl_id;
  ReplaceInstBeforeConstantUse(context, impl.first_owning_decl_id, impl_decl);

  // For an `extend impl` declaration, mark the impl as extending this `impl`.
  if (extend_impl) {
    auto& stored_impl_info = context.impls().Get(impl_decl.impl_id);
    auto self_type_id =
        context.types().GetTypeIdForTypeInstId(stored_impl_info.self_id);
    if (self_type_id != SemIR::ErrorInst::TypeId) {
      auto constraint_id = impl.constraint_id;
      if (stored_impl_info.generic_id.has_value()) {
        constraint_id = AddTypeInst<SemIR::SpecificConstant>(
            context, SemIR::LocId(constraint_id),
            {.type_id = SemIR::TypeType::TypeId,
             .inst_id = constraint_id,
             .specific_id = context.generics().GetSelfSpecific(
                 stored_impl_info.generic_id)});
      }
      if (!ExtendImpl(context, extend_impl->extend_node_id, loc_id,
                      impl_decl.impl_id, extend_impl->self_type_node_id,
                      self_type_id, implicit_params_loc_id, constraint_id,
                      extend_impl->constraint_type_id)) {
        // Don't allow the invalid impl to be used.
        FillImplWitnessWithErrors(context, stored_impl_info);
      }
    }
  }

  // Impl definitions are required in the same file as the declaration. We skip
  // this requirement if we've already issued an invalid redeclaration error, or
  // there is an error that would prevent the impl from being legal to define.
  if (!is_definition) {
    auto& stored_impl = context.impls().Get(impl_id);
    if (stored_impl.witness_id != SemIR::ErrorInst::InstId) {
      context.definitions_required_by_decl().push_back(
          stored_impl.latest_decl_id());
    }
  }

  return {impl_id, impl.latest_decl_id()};
}

}  // namespace Carbon::Check
