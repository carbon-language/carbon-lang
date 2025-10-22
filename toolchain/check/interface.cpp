// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/interface.h"

#include <algorithm>
#include <cstddef>

#include "common/concepts.h"
#include "toolchain/check/context.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/merge.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/type.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto BuildAssociatedEntity(Context& context, SemIR::InterfaceId interface_id,
                           SemIR::InstId decl_id) -> SemIR::InstId {
  auto& interface_info = context.interfaces().Get(interface_id);
  if (!interface_info.is_being_defined()) {
    // This should only happen if the interface is erroneously defined more than
    // once.
    // TODO: Find a way to CHECK this.
    return SemIR::ErrorInst::InstId;
  }

  // This associated entity is being declared as a member of the self specific
  // of the interface.
  auto interface_specific_id =
      context.generics().GetSelfSpecific(interface_info.generic_id);

  // Register this declaration as declaring an associated entity.
  auto index = SemIR::ElementIndex(
      context.args_type_info_stack().PeekCurrentBlockContents().size());
  context.args_type_info_stack().AddInstId(decl_id);

  // Name lookup for the declaration's name should name the associated entity,
  // not the declaration itself.
  auto type_id =
      GetAssociatedEntityType(context, interface_id, interface_specific_id);
  return AddInst<SemIR::AssociatedEntity>(
      context, SemIR::LocId(decl_id),
      {.type_id = type_id, .index = index, .decl_id = decl_id});
}

// Returns the `Self` binding for an interface, given a specific for the
// interface and a generic for an associated entity within it.
static auto GetSelfBinding(Context& context,
                           SemIR::SpecificId interface_specific_id,
                           SemIR::GenericId assoc_entity_generic_id)
    -> SemIR::InstId {
  const auto& generic = context.generics().Get(assoc_entity_generic_id);
  auto bindings = context.inst_blocks().Get(generic.bindings_id);
  auto interface_args_id =
      context.specifics().GetArgsOrEmpty(interface_specific_id);
  auto interface_args = context.inst_blocks().Get(interface_args_id);

  // The `Self` binding is the first binding after the interface's arguments.
  auto self_binding_id = bindings[interface_args.size()];

  // Check that we found the self binding. The binding might be a
  // `BindSymbolicName` or an `ImportRef` naming one.
  auto self_binding_const_inst_id =
      context.constant_values().GetConstantInstId(self_binding_id);
  auto bind_name_inst = context.insts().GetAs<SemIR::BindSymbolicName>(
      self_binding_const_inst_id);
  CARBON_CHECK(
      context.entity_names().Get(bind_name_inst.entity_name_id).name_id ==
          SemIR::NameId::SelfType,
      "Expected a Self binding, found {0}", bind_name_inst);

  return self_binding_id;
}

// Given a `Self` type and a witness that it implements an interface, along with
// that interface's `Self` binding, forms and returns a facet that can be used
// as the argument for that `Self` binding.
static auto GetSelfFacet(Context& context,
                         SemIR::SpecificId interface_specific_id,
                         SemIR::GenericId generic_id,
                         SemIR::TypeId self_type_id,
                         SemIR::InstId self_witness_id) -> SemIR::InstId {
  auto self_binding_id =
      GetSelfBinding(context, interface_specific_id, generic_id);
  auto self_facet_type_id = SemIR::GetTypeOfInstInSpecific(
      context.sem_ir(), interface_specific_id, self_binding_id);
  // Create a facet value to be the value of `Self` in the interface.
  // TODO: Pass this in instead of creating it here. The caller sometimes
  // already has a facet value.
  auto type_inst_id = context.types().GetInstId(self_type_id);
  auto witnesses_block_id =
      context.inst_blocks().AddCanonical({self_witness_id});
  auto self_value_const_id = TryEvalInst(
      context, SemIR::FacetValue{.type_id = self_facet_type_id,
                                 .type_inst_id = type_inst_id,
                                 .witnesses_block_id = witnesses_block_id});
  return context.constant_values().GetInstId(self_value_const_id);
}

// Builds and returns the argument list from `interface_specific_id` with a
// value for the `Self` parameter of `generic_id` appended.
static auto GetGenericArgsWithSelfType(Context& context,
                                       SemIR::SpecificId interface_specific_id,
                                       SemIR::GenericId generic_id,
                                       SemIR::TypeId self_type_id,
                                       SemIR::InstId witness_inst_id,
                                       std::size_t reserve_args_size = 0)
    -> llvm::SmallVector<SemIR::InstId> {
  auto interface_args_id =
      context.specifics().GetArgsOrEmpty(interface_specific_id);
  auto interface_args = context.inst_blocks().Get(interface_args_id);

  llvm::SmallVector<SemIR::InstId> arg_ids;
  arg_ids.reserve(std::max(reserve_args_size, interface_args.size() + 1));

  // Start with the enclosing arguments from the interface.
  arg_ids.assign(interface_args.begin(), interface_args.end());

  // Add the `Self` argument.
  arg_ids.push_back(GetSelfFacet(context, interface_specific_id, generic_id,
                                 self_type_id, witness_inst_id));

  return arg_ids;
}

auto GetSelfSpecificForInterfaceMemberWithSelfType(
    Context& context, SemIR::LocId loc_id,
    SemIR::SpecificId interface_specific_id, SemIR::GenericId generic_id,
    SemIR::SpecificId enclosing_specific_id, SemIR::TypeId self_type_id,
    SemIR::InstId witness_inst_id) -> SemIR::SpecificId {
  const auto& generic = context.generics().Get(generic_id);
  auto self_specific_args = context.inst_blocks().Get(
      context.specifics().Get(generic.self_specific_id).args_id);

  auto arg_ids = GetGenericArgsWithSelfType(
      context, interface_specific_id, generic_id, self_type_id, witness_inst_id,
      self_specific_args.size());

  // Determine the number of specific arguments that enclose the point where
  // this self specific will be used from. In an impl, this will be the number
  // of parameters that the impl has.
  int num_enclosing_specific_args =
      context.inst_blocks()
          .Get(context.specifics().GetArgsOrEmpty(enclosing_specific_id))
          .size();
  // The index of each remaining generic parameter is adjusted to match the
  // numbering at the point where the self specific is used.
  int index_delta = num_enclosing_specific_args - arg_ids.size();

  // Take any trailing argument values from the self specific.
  // TODO: If these refer to outer arguments, for example in their types, we may
  // need to perform extra substitutions here.
  for (auto arg_id : self_specific_args.drop_front(arg_ids.size())) {
    auto new_arg_id = context.constant_values().GetConstantInstId(arg_id);
    if (index_delta) {
      // If this parameter would have a new index in the context described by
      // `enclosing_specific_id`, form a new binding with an adjusted index.
      auto bind_name = context.insts().GetAs<SemIR::BindSymbolicName>(
          context.constant_values().GetConstantInstId(arg_id));
      auto entity_name = context.entity_names().Get(bind_name.entity_name_id);
      entity_name.bind_index_value += index_delta;
      CARBON_CHECK(entity_name.bind_index_value >= 0);
      bind_name.entity_name_id =
          context.entity_names().AddCanonical(entity_name);
      new_arg_id =
          context.constant_values().GetInstId(TryEvalInst(context, bind_name));
    }
    arg_ids.push_back(new_arg_id);
  }

  return MakeSpecific(context, loc_id, generic_id, arg_ids);
}

auto GetTypeForSpecificAssociatedEntity(Context& context, SemIR::LocId loc_id,
                                        SemIR::SpecificId interface_specific_id,
                                        SemIR::InstId decl_id,
                                        SemIR::TypeId self_type_id,
                                        SemIR::InstId self_witness_id)
    -> SemIR::TypeId {
  auto decl_constant_inst_id =
      context.constant_values().GetConstantInstId(decl_id);
  if (decl_constant_inst_id == SemIR::ErrorInst::InstId) {
    return SemIR::ErrorInst::TypeId;
  }

  auto decl = context.insts().Get(decl_constant_inst_id);
  if (auto assoc_const = decl.TryAs<SemIR::AssociatedConstantDecl>()) {
    // Form a specific for the associated constant, and grab the type from
    // there.
    auto generic_id = context.associated_constants()
                          .Get(assoc_const->assoc_const_id)
                          .generic_id;
    auto arg_ids =
        GetGenericArgsWithSelfType(context, interface_specific_id, generic_id,
                                   self_type_id, self_witness_id);
    auto const_specific_id = MakeSpecific(context, loc_id, generic_id, arg_ids);
    return SemIR::GetTypeOfInstInSpecific(context.sem_ir(), const_specific_id,
                                          decl_id);
  }

  if (auto fn = context.types().TryGetAs<SemIR::FunctionType>(decl.type_id())) {
    // Form the type of the function within the interface, and attach the `Self`
    // type.
    auto interface_fn_type_id = SemIR::GetTypeOfInstInSpecific(
        context.sem_ir(), interface_specific_id, decl_id);
    auto self_facet_id =
        GetSelfFacet(context, interface_specific_id,
                     context.functions().Get(fn->function_id).generic_id,
                     self_type_id, self_witness_id);
    return GetFunctionTypeWithSelfType(
        context, context.types().GetInstId(interface_fn_type_id),
        self_facet_id);
  }

  CARBON_FATAL("Unexpected kind for associated constant {0}", decl);
}

auto AddSelfGenericParameter(Context& context, SemIR::TypeId type_id,
                             SemIR::NameScopeId scope_id, bool is_template)
    -> SemIR::InstId {
  auto entity_name_id = context.entity_names().AddSymbolicBindingName(
      SemIR::NameId::SelfType, scope_id,
      context.scope_stack().AddCompileTimeBinding(), is_template);
  // Because there is no equivalent non-symbolic value, we use `None` as
  // the `value_id` on the `BindSymbolicName`.
  auto self_param_inst_id =
      AddInst(context, SemIR::LocIdAndInst::NoLoc<SemIR::BindSymbolicName>(
                           {.type_id = type_id,
                            .entity_name_id = entity_name_id,
                            .value_id = SemIR::InstId::None}));
  context.scope_stack().PushCompileTimeBinding(self_param_inst_id);
  context.name_scopes().AddRequiredName(scope_id, SemIR::NameId::SelfType,
                                        self_param_inst_id);
  return self_param_inst_id;
}

template <typename EntityT>
  requires std::same_as<EntityT, SemIR::Interface>
static auto TryGetEntity(Context& context, SemIR::Inst inst)
    -> const SemIR::EntityWithParamsBase* {
  if (auto decl = inst.TryAs<SemIR::InterfaceDecl>()) {
    return &context.interfaces().Get(decl->interface_id);
  } else {
    return nullptr;
  }
}

template <typename EntityT>
  requires std::same_as<EntityT, SemIR::NamedConstraint>
static auto TryGetEntity(Context& context, SemIR::Inst inst)
    -> const SemIR::EntityWithParamsBase* {
  if (auto decl = inst.TryAs<SemIR::NamedConstraintDecl>()) {
    return &context.named_constraints().Get(decl->named_constraint_id);
  } else {
    return nullptr;
  }
}

template <typename EntityT>
  requires std::same_as<EntityT, SemIR::Interface>
static constexpr auto DeclTokenKind() -> Lex::TokenKind {
  return Lex::TokenKind::Interface;
}

template <typename EntityT>
  requires std::same_as<EntityT, SemIR::NamedConstraint>
static constexpr auto DeclTokenKind() -> Lex::TokenKind {
  return Lex::TokenKind::Constraint;
}

template <typename EntityT>
  requires SameAsOneOf<EntityT, SemIR::Interface, SemIR::NamedConstraint>
auto TryGetExistingDecl(Context& context, const NameComponent& name,
                        SemIR::ScopeLookupResult lookup_result,
                        const EntityT& entity, bool is_definition)
    -> std::optional<SemIR::Inst> {
  if (lookup_result.is_poisoned()) {
    // This is a declaration of a poisoned name.
    DiagnosePoisonedName(context, name.name_id,
                         lookup_result.poisoning_loc_id(), name.name_loc_id);
    return std::nullopt;
  }

  if (!lookup_result.is_found()) {
    return std::nullopt;
  }

  SemIR::InstId existing_id = lookup_result.target_inst_id();
  SemIR::Inst existing_decl_inst = context.insts().Get(existing_id);
  const auto* existing_decl_entity =
      TryGetEntity<EntityT>(context, existing_decl_inst);
  if (!existing_decl_entity) {
    // This is a redeclaration with a different entity kind.
    DiagnoseDuplicateName(context, name.name_id, name.name_loc_id,
                          SemIR::LocId(existing_id));
    return std::nullopt;
  }

  if (!CheckRedeclParamsMatch(
          context,
          DeclParams(SemIR::LocId(entity.latest_decl_id()),
                     name.first_param_node_id, name.last_param_node_id,
                     name.implicit_param_patterns_id, name.param_patterns_id),
          DeclParams(*existing_decl_entity))) {
    // Mismatch is diagnosed already if found.
    return std::nullopt;
  }

  // TODO: This should be refactored a little, particularly for
  // prev_import_ir_id. See similar logic for classes and functions, which
  // might also be refactored to merge.
  DiagnoseIfInvalidRedecl(
      context, DeclTokenKind<EntityT>(), existing_decl_entity->name_id,
      RedeclInfo(entity, SemIR::LocId(entity.latest_decl_id()), is_definition),
      RedeclInfo(*existing_decl_entity,
                 SemIR::LocId(existing_decl_entity->latest_decl_id()),
                 existing_decl_entity->has_definition_started()),
      /*prev_import_ir_id=*/SemIR::ImportIRId::None);

  if (is_definition && existing_decl_entity->has_definition_started()) {
    // DiagnoseIfInvalidRedecl would diagnose an error in this case, since we'd
    // have two definitions. Given the declaration parts of the definitions
    // match, we would be able to use the prior declaration for error recovery,
    // except that having two definitions causes larger problems for generics.
    // All interfaces (and named constraints) are generic with an implicit Self
    // compile time binding.
    return std::nullopt;
  }

  // This is a matching redeclaration of an existing entity of the same type.
  return existing_decl_inst;
}

template auto TryGetExistingDecl(Context& context, const NameComponent& name,
                                 SemIR::ScopeLookupResult lookup_result,
                                 const SemIR::Interface& entity,
                                 bool is_definition)
    -> std::optional<SemIR::Inst>;
template auto TryGetExistingDecl(Context& context, const NameComponent& name,
                                 SemIR::ScopeLookupResult lookup_result,
                                 const SemIR::NamedConstraint& entity,
                                 bool is_definition)
    -> std::optional<SemIR::Inst>;

}  // namespace Carbon::Check
