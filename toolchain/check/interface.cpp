// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/interface.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <type_traits>

#include "common/concepts.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/core_identifier.h"
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

  // This associated entity is being declared as a member of an interface. We
  // use the self-specific of the interface-without-self as the AssociatedEntity
  // names the externally facing SpecificInterface (without self).
  auto interface_without_self_specific_id =
      context.generics().GetSelfSpecific(interface_info.generic_id);

  // Register this declaration as declaring an associated entity.
  auto index = SemIR::ElementIndex(
      context.args_type_info_stack().PeekCurrentBlockContents().size());
  context.args_type_info_stack().AddInstId(decl_id);

  // Name lookup for the declaration's name should name the associated entity,
  // not the declaration itself.
  auto type_id = GetAssociatedEntityType(context, interface_id,
                                         interface_without_self_specific_id);
  return AddInst<SemIR::AssociatedEntity>(
      context, SemIR::LocId(decl_id),
      {.type_id = type_id, .index = index, .decl_id = decl_id});
}

auto GetSelfSpecificForInterfaceMemberWithSelfType(
    Context& context, SemIR::LocId loc_id,
    SemIR::SpecificId interface_with_self_specific_id,
    SemIR::GenericId generic_id, SemIR::SpecificId enclosing_specific_id)
    -> SemIR::SpecificId {
  const auto& generic = context.generics().Get(generic_id);
  auto self_specific_args = context.inst_blocks().Get(
      context.specifics().Get(generic.self_specific_id).args_id);

  auto arg_ids = llvm::SmallVector<SemIR::InstId>(context.inst_blocks().Get(
      context.specifics().GetArgsOrEmpty(interface_with_self_specific_id)));

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
      auto bind_name = context.insts().GetAs<SemIR::SymbolicBinding>(
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

auto GetTypeForSpecificAssociatedEntity(
    Context& context, SemIR::SpecificId interface_with_self_specific_id,
    SemIR::InstId decl_id) -> SemIR::TypeId {
  auto decl_constant_inst_id =
      context.constant_values().GetConstantInstId(decl_id);
  if (decl_constant_inst_id == SemIR::ErrorInst::InstId) {
    return SemIR::ErrorInst::TypeId;
  }

  auto decl = context.insts().Get(decl_constant_inst_id);
  if (auto assoc_const = decl.TryAs<SemIR::AssociatedConstantDecl>()) {
    return SemIR::GetTypeOfInstInSpecific(
        context.sem_ir(), interface_with_self_specific_id, decl_id);
  }

  if (auto fn = context.types().TryGetAs<SemIR::FunctionType>(decl.type_id())) {
    // Form the type of the function within the interface, and attach the `Self`
    // type.
    auto interface_fn_type_id = SemIR::GetTypeOfInstInSpecific(
        context.sem_ir(), interface_with_self_specific_id, decl_id);
    auto self_facet_id = context.inst_blocks()
                             .Get(context.specifics().GetArgsOrEmpty(
                                 interface_with_self_specific_id))
                             .back();
    return GetFunctionTypeWithSelfType(
        context, context.types().GetTypeInstId(interface_fn_type_id),
        self_facet_id);
  }

  CARBON_FATAL("Unexpected kind for associated constant {0}", decl);
}

auto AddSelfSymbolicBindingToScope(Context& context,
                                   SemIR::LocId definition_loc_id,
                                   SemIR::TypeId type_id,
                                   SemIR::NameScopeId scope_id,
                                   bool is_template) -> SemIR::InstId {
  auto entity_name_id = context.entity_names().AddSymbolicBindingName(
      SemIR::NameId::SelfType, scope_id,
      context.scope_stack().AddCompileTimeBinding(), is_template,
      /*is_unused=*/false, /*is_frozen_period_self=*/false);
  // Because there is no equivalent non-symbolic value, we use `None` as
  // the `value_id` on the `SymbolicBinding`.
  auto self_param_inst_id =
      AddInst<SemIR::SymbolicBinding>(context, definition_loc_id,
                                      {.type_id = type_id,
                                       .entity_name_id = entity_name_id,
                                       .value_id = SemIR::InstId::None});
  context.name_scopes().AddRequiredName(scope_id, SemIR::NameId::SelfType,
                                        self_param_inst_id);
  return self_param_inst_id;
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
  using EntityIdT =
      std::conditional_t<std::is_same_v<EntityT, SemIR::Interface>,
                         SemIR::InterfaceId, SemIR::NamedConstraintId>;
  constexpr bool IsInterface = std::is_same_v<EntityIdT, SemIR::InterfaceId>;

  if (lookup_result.is_poisoned()) {
    // This is a declaration of a poisoned name.
    DiagnosePoisonedName(context, name.name_id,
                         lookup_result.poisoning_loc_id(), name.name_loc_id);
    return std::nullopt;
  }

  if (!lookup_result.is_found()) {
    return std::nullopt;
  }

  auto prev_id = lookup_result.target_inst_id();
  auto prev = context.insts().Get(prev_id);

  auto prev_entity_id = EntityIdT::None;
  auto prev_import_ir_id = SemIR::ImportIRId::None;
  auto existing_decl_id = SemIR::InstId::None;
  CARBON_KIND_SWITCH(prev) {
    case CARBON_KIND(SemIR::InterfaceDecl interface_decl): {
      if constexpr (IsInterface) {
        prev_entity_id = interface_decl.interface_id;
        existing_decl_id = prev_id;
      }
      break;
    }
    case CARBON_KIND(SemIR::NamedConstraintDecl named_constraint_decl): {
      if constexpr (!IsInterface) {
        prev_entity_id = named_constraint_decl.named_constraint_id;
        existing_decl_id = prev_id;
      }
      break;
    }
    case CARBON_KIND(SemIR::ImportRefLoaded import_ref): {
      auto import_ir_inst =
          context.import_ir_insts().Get(import_ref.import_ir_inst_id);

      // Verify the decl so that things like aliases are name conflicts.
      const auto* import_ir =
          context.import_irs().Get(import_ir_inst.ir_id()).sem_ir;
      if constexpr (IsInterface) {
        if (!import_ir->insts().Is<SemIR::InterfaceDecl>(
                import_ir_inst.inst_id())) {
          break;
        }
      } else {
        if (!import_ir->insts().Is<SemIR::NamedConstraintDecl>(
                import_ir_inst.inst_id())) {
          break;
        }
      }

      // Use the constant value to get the ID.
      auto decl_value = context.insts().Get(
          context.constant_values().GetConstantInstId(prev_id));
      if (auto facet_type = decl_value.TryAs<SemIR::FacetType>()) {
        auto declared_facet_type = context.declared_facet_types().Get(
            facet_type->declared_facet_type_id);
        if constexpr (IsInterface) {
          prev_entity_id =
              declared_facet_type.extend_constraints[0].interface_id;
        } else {
          prev_entity_id = declared_facet_type.extend_named_constraints[0]
                               .named_constraint_id;
        }
        prev_import_ir_id = import_ir_inst.ir_id();
      }
      break;
    }
    default:
      break;
  }

  if (!prev_entity_id.has_value()) {
    // This is a redeclaration with a different entity kind.
    DiagnoseDuplicateName(context, name.name_id, name.name_loc_id,
                          SemIR::LocId(prev_id));
    return std::nullopt;
  }

  auto& prev_entity = [&]() -> EntityT& {
    if constexpr (IsInterface) {
      return context.interfaces().Get(prev_entity_id);
    } else {
      return context.named_constraints().Get(prev_entity_id);
    }
  }();

  if (!CheckRedeclParamsMatch(
          context,
          DeclParams(SemIR::LocId(entity.latest_decl_id()),
                     name.first_param_node_id, name.last_param_node_id,
                     name.implicit_param_patterns_id, name.param_patterns_id),
          DeclParams(prev_entity))) {
    // Mismatch is diagnosed already if found.
    return std::nullopt;
  }

  DiagnoseIfInvalidRedecl(
      context, DeclTokenKind<EntityT>(), prev_entity.name_id,
      RedeclInfo(entity, SemIR::LocId(entity.latest_decl_id()), is_definition),
      RedeclInfo(prev_entity, SemIR::LocId(prev_entity.latest_decl_id()),
                 prev_entity.has_definition_started()),
      prev_import_ir_id);

  if (is_definition && prev_entity.has_definition_started()) {
    // DiagnoseIfInvalidRedecl would diagnose an error in this case, since we'd
    // have two definitions. Given the declaration parts of the definitions
    // match, we would be able to use the prior declaration for error recovery,
    // except that having two definitions causes larger problems for generics.
    // All interfaces (and named constraints) are generic with an implicit Self
    // compile time binding.
    return std::nullopt;
  }

  if (is_definition) {
    prev_entity.MergeDefinition(entity);
  }

  if (prev_import_ir_id.has_value()) {
    prev_entity.first_owning_decl_id = entity.first_owning_decl_id;
    ReplacePrevInstForMerge(context, entity.parent_scope_id,
                            prev_entity.name_id, entity.first_owning_decl_id);
  }

  // This is a matching redeclaration of an existing entity of the same type.
  return existing_decl_id.has_value() ? std::optional<SemIR::Inst>(prev)
                                      : std::nullopt;
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
