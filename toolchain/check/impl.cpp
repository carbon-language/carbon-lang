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
#include "toolchain/check/name_scope.h"
#include "toolchain/check/thunk.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/check/type_structure.h"
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

auto CheckAssociatedFunctionImplementation(
    Context& context, SemIR::FunctionType interface_function_type,
    SemIR::InstId impl_decl_id, SemIR::TypeId self_type_id,
    SemIR::InstId witness_inst_id, bool defer_thunk_definition)
    -> SemIR::InstId {
  auto impl_function_decl =
      context.insts().TryGetAs<SemIR::FunctionDecl>(impl_decl_id);
  if (!impl_function_decl) {
    if (impl_decl_id != SemIR::ErrorInst::InstId) {
      CARBON_DIAGNOSTIC(ImplFunctionWithNonFunction, Error,
                        "associated function {0} implemented by non-function",
                        SemIR::NameId);
      auto builder = context.emitter().Build(
          impl_decl_id, ImplFunctionWithNonFunction,
          context.functions().Get(interface_function_type.function_id).name_id);
      NoteAssociatedFunction(context, builder,
                             interface_function_type.function_id);
      builder.Emit();
    }

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
                    interface_function_specific_id, impl_decl_id,
                    defer_thunk_definition);
}

// Builds an initial witness from the rewrites in the facet type, if any.
auto ImplWitnessForDeclaration(Context& context, const SemIR::Impl& impl)
    -> SemIR::InstId {
  CARBON_CHECK(!impl.has_definition_started());

  auto self_type_id = context.types().GetTypeIdForTypeInstId(impl.self_id);
  if (self_type_id == SemIR::ErrorInst::TypeId) {
    // When 'impl as' is invalid, the self type is an error.
    return SemIR::ErrorInst::InstId;
  }

  return InitialFacetTypeImplWitness(
      context, SemIR::LocId(impl.latest_decl_id()), impl.constraint_id,
      impl.self_id, impl.interface,
      context.generics().GetSelfSpecific(impl.generic_id));
}

auto ImplWitnessStartDefinition(Context& context, SemIR::Impl& impl) -> void {
  CARBON_CHECK(impl.is_being_defined());
  CARBON_CHECK(impl.witness_id.has_value());
  if (impl.witness_id == SemIR::ErrorInst::InstId) {
    return;
  }

  if (!RequireCompleteType(
          context, context.types().GetTypeIdForTypeInstId(impl.constraint_id),
          SemIR::LocId(impl.constraint_id), [&] {
            CARBON_DIAGNOSTIC(ImplAsIncompleteFacetTypeDefinition, Error,
                              "definition of impl as incomplete facet type {0}",
                              InstIdAsType);
            return context.emitter().Build(SemIR::LocId(impl.latest_decl_id()),
                                           ImplAsIncompleteFacetTypeDefinition,
                                           impl.constraint_id);
          })) {
    FillImplWitnessWithErrors(context, impl);
    return;
  }

  const auto& interface = context.interfaces().Get(impl.interface.interface_id);

  auto assoc_entities =
      context.inst_blocks().Get(interface.associated_entities_id);
  for (auto decl_id : assoc_entities) {
    LoadImportRef(context, decl_id);
  }

  auto witness = context.insts().GetAs<SemIR::ImplWitness>(impl.witness_id);
  auto witness_table =
      context.insts().GetAs<SemIR::ImplWitnessTable>(witness.witness_table_id);
  auto witness_block =
      context.inst_blocks().GetMutable(witness_table.elements_id);

  // The impl declaration may have created a placeholder witness table, or a
  // full witness table. We can detect that the witness table is a placeholder
  // table if it's not the `Empty` id, but it is empty still. If it was a
  // placeholder, we can replace the placeholder here with a table of the proper
  // size, since the interface must be complete for the impl definition.
  bool witness_table_is_placeholder =
      witness_table.elements_id != SemIR::InstBlockId::Empty &&
      witness_block.empty();
  if (witness_table_is_placeholder) {
    // TODO: Since our `empty_table` repeats the same value throughout, we could
    // skip an allocation here if there was a `ReplacePlaceholder` function that
    // took a size and value instead of an array of values.
    llvm::SmallVector<SemIR::InstId> empty_table(
        assoc_entities.size(), SemIR::InstId::ImplWitnessTablePlaceholder);
    context.inst_blocks().ReplacePlaceholder(witness_table.elements_id,
                                             empty_table);
    witness_block = context.inst_blocks().GetMutable(witness_table.elements_id);
  }

  // Check we have a value for all non-function associated constants in the
  // witness.
  for (auto [assoc_entity, witness_value] :
       llvm::zip_equal(assoc_entities, witness_block)) {
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
       llvm::zip_equal(assoc_entities, witness_block)) {
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
              impl.witness_id, /*defer_thunk_definition=*/true);
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

  auto identified_id = RequireIdentifiedFacetType(
      context, SemIR::LocId(constraint_id), *facet_type, [&] {
        CARBON_DIAGNOSTIC(ImplOfUnidentifiedFacetType, Error,
                          "facet type {0} cannot be identified in `impl as`",
                          InstIdAsType);
        return context.emitter().Build(
            impl_decl_id, ImplOfUnidentifiedFacetType, constraint_id);
      });
  if (!identified_id.has_value()) {
    return SemIR::SpecificInterface::None;
  }
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
static auto CheckImplRedeclParamsMatch(Context& context,
                                       const SemIR::Impl& new_impl,
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
static auto IsValidImplRedecl(Context& context, const SemIR::Impl& new_impl,
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

// Sets the `ImplId` in the `ImplWitnessTable`.
static auto AssignImplIdInWitness(Context& context, SemIR::ImplId impl_id,
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

// Looks for any unused generic bindings. If one is found, it is diagnosed and
// false is returned.
static auto VerifyAllGenericBindingsUsed(Context& context, SemIR::LocId loc_id,
                                         SemIR::LocId implicit_params_loc_id,
                                         SemIR::Impl& impl) -> bool {
  if (impl.witness_id == SemIR::ErrorInst::InstId) {
    return true;
  }
  if (!impl.generic_id.has_value()) {
    return true;
  }

  if (impl.implicit_param_patterns_id.has_value()) {
    for (auto inst_id :
         context.inst_blocks().Get(impl.implicit_param_patterns_id)) {
      if (inst_id == SemIR::ErrorInst::InstId) {
        // An error was already diagnosed for a generic binding.
        return true;
      }
    }
  }

  auto deduced_specific_id = DeduceImplArguments(
      context, loc_id, impl, context.constant_values().Get(impl.self_id),
      impl.interface.specific_id);
  if (deduced_specific_id.has_value()) {
    // Deduction succeeded, all bindings were used.
    return true;
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
  return false;
}

// Apply an `extend impl` declaration by extending the parent scope with the
// `impl`. If there's an error it is diagnosed and false is returned.
static auto ApplyExtendImplAs(Context& context, SemIR::LocId loc_id,
                              const SemIR::Impl& impl,
                              Parse::NodeId extend_node,
                              SemIR::LocId implicit_params_loc_id) -> bool {
  auto parent_scope_id = context.decl_name_stack().PeekParentScopeId();

  // TODO: Also handle the parent scope being a mixin or an interface.
  auto class_scope = TryAsClassScope(context, parent_scope_id);
  if (!class_scope) {
    if (impl.witness_id != SemIR::ErrorInst::InstId) {
      CARBON_DIAGNOSTIC(
          ExtendImplOutsideClass, Error,
          "`extend impl` can only be used in an interface or class");
      context.emitter().Emit(loc_id, ExtendImplOutsideClass);
    }
    return false;
  }

  auto& parent_scope = *class_scope->name_scope;

  // An error was already diagnosed, but this is `extend impl as` inside a
  // class, so propagate the error into the enclosing class scope.
  if (impl.witness_id == SemIR::ErrorInst::InstId) {
    parent_scope.set_has_error();
    return false;
  }

  if (implicit_params_loc_id.has_value()) {
    CARBON_DIAGNOSTIC(ExtendImplForall, Error,
                      "cannot `extend` a parameterized `impl`");
    context.emitter().Emit(extend_node, ExtendImplForall);
    parent_scope.set_has_error();
    return false;
  }

  if (!RequireCompleteType(
          context, context.types().GetTypeIdForTypeInstId(impl.constraint_id),
          SemIR::LocId(impl.constraint_id), [&] {
            CARBON_DIAGNOSTIC(ExtendImplAsIncomplete, Error,
                              "`extend impl as` incomplete facet type {0}",
                              InstIdAsType);
            return context.emitter().Build(impl.latest_decl_id(),
                                           ExtendImplAsIncomplete,
                                           impl.constraint_id);
          })) {
    parent_scope.set_has_error();
    return false;
  }

  if (!impl.generic_id.has_value()) {
    parent_scope.AddExtendedScope(impl.constraint_id);
  } else {
    auto constraint_id_in_self_specific = AddTypeInst<SemIR::SpecificConstant>(
        context, SemIR::LocId(impl.constraint_id),
        {.type_id = SemIR::TypeType::TypeId,
         .inst_id = impl.constraint_id,
         .specific_id = context.generics().GetSelfSpecific(impl.generic_id)});
    parent_scope.AddExtendedScope(constraint_id_in_self_specific);
  }
  return true;
}

auto GetOrAddImpl(Context& context, SemIR::LocId loc_id,
                  SemIR::LocId implicit_params_loc_id, SemIR::Impl impl,
                  Parse::NodeId extend_node) -> SemIR::ImplId {
  auto impl_id = SemIR::ImplId::None;

  // Look for an existing matching declaration.
  auto lookup_bucket_ref = context.impls().GetOrAddLookupBucket(impl);
  // TODO: Detect two impl declarations with the same self type and interface,
  // and issue an error if they don't match.
  for (auto prev_impl_id : lookup_bucket_ref) {
    if (CheckImplRedeclParamsMatch(context, impl, prev_impl_id)) {
      if (IsValidImplRedecl(context, impl, prev_impl_id)) {
        impl_id = prev_impl_id;
      } else {
        // IsValidImplRedecl() has issued a diagnostic, take care to avoid
        // generating more diagnostics for this declaration.
        impl.witness_id = SemIR::ErrorInst::InstId;
      }
      break;
    }
  }

  if (impl_id.has_value()) {
    // This is a redeclaration of another impl, now held in `impl_id`.
    auto& prev_impl = context.impls().Get(impl_id);
    FinishGenericRedecl(context, prev_impl.generic_id);
    return impl_id;
  }

  // This is a new declaration (possibly with an attached definition). Create a
  // new `impl_id`, filling the missing generic and witness in the provided
  // `Impl`.

  impl.generic_id = BuildGeneric(context, impl.latest_decl_id());

  // Due to lack of an instruction to set to `ErrorInst`, an `InterfaceId::None`
  // indicates that the interface could not be identified and an error was
  // diagnosed. If there's any error in the construction of the impl, then the
  // witness can't be constructed. We set it to `ErrorInst` to make the impl
  // unusable for impl lookup.
  if (!impl.interface.interface_id.has_value() ||
      impl.self_id == SemIR::ErrorInst::TypeInstId ||
      impl.constraint_id == SemIR::ErrorInst::TypeInstId) {
    impl.witness_id = SemIR::ErrorInst::InstId;
    // TODO: We might also want to mark that the name scope for the impl has an
    // error -- at least once we start making name lookups within the impl also
    // look into the facet (eg, so you can name associated constants from within
    // the impl).
  }

  if (impl.witness_id != SemIR::ErrorInst::InstId) {
    // This makes either a placeholder witness or a full witness table. The full
    // witness table is deferred to the impl definition unless the declaration
    // uses rewrite constraints to set values of associated constants in the
    // interface.
    impl.witness_id = ImplWitnessForDeclaration(context, impl);
  }

  FinishGenericDecl(context, SemIR::LocId(impl.latest_decl_id()),
                    impl.generic_id);
  // From here on, use the `Impl` from the `ImplStore` instead of `impl` in
  // order to make and see any changes to the `Impl`.
  impl_id = context.impls().Add(impl);
  lookup_bucket_ref.push_back(impl_id);
  AssignImplIdInWitness(context, impl_id, impl.witness_id);

  auto& stored_impl = context.impls().Get(impl_id);

  // Look to see if there are any generic bindings on the `impl` declaration
  // that are not deducible. If so, and the `impl` does not actually use all its
  // generic bindings, and will never be matched. This should be diagnosed to
  // the user.
  if (!VerifyAllGenericBindingsUsed(context, loc_id, implicit_params_loc_id,
                                    stored_impl)) {
    FillImplWitnessWithErrors(context, stored_impl);
  }

  if (extend_node.has_value()) {
    if (!ApplyExtendImplAs(context, loc_id, stored_impl, extend_node,
                           implicit_params_loc_id)) {
      FillImplWitnessWithErrors(context, stored_impl);
    }
  }

  return impl_id;
}

}  // namespace Carbon::Check
