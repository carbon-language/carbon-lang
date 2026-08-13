// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_MERGE_H_
#define CARBON_TOOLCHAIN_CHECK_MERGE_H_

#include <optional>
#include <type_traits>

#include "common/check.h"
#include "common/concepts.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/function.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/subst.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/sem_ir/class.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/import_ir.h"
#include "toolchain/sem_ir/interface.h"
#include "toolchain/sem_ir/named_constraint.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Diagnoses an `extern` declaration that was not preceded by a declaration in
// the API file.
auto DiagnoseExternRequiresDeclInApiFile(Context& context, SemIR::LocId loc_id)
    -> void;

// Information on new and previous declarations for DiagnoseIfInvalidRedecl.
struct RedeclInfo {
  explicit RedeclInfo(const SemIR::EntityWithParamsBase& params,
                      SemIR::LocId loc_id, bool is_definition)
      : loc_id(loc_id),
        is_definition(is_definition),
        is_extern(params.is_extern),
        extern_library_id(params.extern_library_id) {}

  // The associated diagnostic location.
  SemIR::LocId loc_id;
  // True if a definition.
  bool is_definition;
  // True if an `extern` declaration.
  bool is_extern;
  // The library name in `extern library`, or `None` if not present.
  SemIR::LibraryNameId extern_library_id;
};

// Checks for various invalid redeclarations. This can emit diagnostics.
// However, merging is still often appropriate for error recovery, so this
// doesn't return whether a diagnostic occurred.
//
// The kinds of things this verifies are:
// - A declaration is not redundant.
// - A definition doesn't redefine a prior definition.
// - The use of `extern` is consistent within a library.
// - Multiple libraries do not declare non-`extern`.
auto DiagnoseIfInvalidRedecl(Context& context, Lex::TokenKind decl_kind,
                             SemIR::NameId name_id, RedeclInfo new_decl,
                             RedeclInfo prev_decl,
                             SemIR::ImportIRId prev_import_ir_id) -> void;

// When the prior name lookup result is an import and we are successfully
// merging, replace the name lookup result with the reference in the current
// file.
auto ReplacePrevInstForMerge(Context& context, SemIR::NameScopeId scope_id,
                             SemIR::NameId name_id, SemIR::InstId new_inst_id)
    -> void;

// Information about the parameters of a declaration, which is common across
// different kinds of entity such as classes and functions.
struct DeclParams {
  explicit DeclParams(const SemIR::EntityWithParamsBase& base)
      : loc_id(base.latest_decl_id()),
        first_param_node_id(base.first_param_node_id),
        last_param_node_id(base.last_param_node_id),
        implicit_param_patterns_id(base.implicit_param_patterns_id),
        param_patterns_id(base.param_patterns_id) {}

  DeclParams(SemIR::LocId loc_id, Parse::NodeId first_param_node_id,
             Parse::NodeId last_param_node_id,
             SemIR::InstBlockId implicit_param_patterns_id,
             SemIR::InstBlockId param_patterns_id)
      : loc_id(loc_id),
        first_param_node_id(first_param_node_id),
        last_param_node_id(last_param_node_id),
        implicit_param_patterns_id(implicit_param_patterns_id),
        param_patterns_id(param_patterns_id) {}

  // The location of the declaration of the entity.
  SemIR::LocId loc_id;

  // Parse tree bounds for the parameters, including both implicit and explicit
  // parameters. These will be compared to match between declaration and
  // definition.
  Parse::NodeId first_param_node_id;
  Parse::NodeId last_param_node_id;

  // The implicit parameters of the entity. Can be `None` if there is no
  // implicit parameter list.
  SemIR::InstBlockId implicit_param_patterns_id;
  // The explicit parameters of the entity. Can be `None` if there is no
  // explicit parameter list.
  SemIR::InstBlockId param_patterns_id;
};

// Checks that the parameters in a redeclaration of an entity match the
// parameters in the prior declaration. If not, produces a diagnostic if
// `diagnose` is true, and returns false.
auto CheckRedeclParamsMatch(Context& context, const DeclParams& new_entity,
                            const DeclParams& prev_entity,
                            SemIR::SpecificId prev_specific_id, bool diagnose,
                            bool check_syntax) -> bool;

inline auto CheckRedeclParamsMatch(Context& context,
                                   const DeclParams& new_entity,
                                   const DeclParams& prev_entity) -> bool {
  return CheckRedeclParamsMatch(context, new_entity, prev_entity,
                                SemIR::SpecificId::None, /*diagnose=*/true,
                                /*check_syntax=*/true);
}

// Fills the previous class id, type id, and import ir id.
inline auto FillPrevEntityInfo(Context& context,
                               const SemIR::ImportIRInst& import_ir_inst,
                               SemIR::Inst decl_val,
                               SemIR::ClassId& prev_entity_id,
                               SemIR::TypeId& prev_type_id,
                               SemIR::ImportIRId& prev_import_ir_id) -> void {
  // Verify the decl so that things like aliases are name conflicts.
  const auto* import_ir =
      context.import_irs().Get(import_ir_inst.ir_id()).sem_ir;
  if (!import_ir->insts().Is<SemIR::ClassDecl>(import_ir_inst.inst_id())) {
    return;
  }

  if (auto class_type = decl_val.TryAs<SemIR::ClassType>()) {
    prev_entity_id = class_type->class_id;
    prev_type_id = SemIR::TypeId::None;
    prev_import_ir_id = import_ir_inst.ir_id();
  } else if (auto generic_class_type =
                 context.types().TryGetAs<SemIR::GenericClassType>(
                     decl_val.type_id())) {
    prev_entity_id = generic_class_type->class_id;
    prev_type_id = SemIR::TypeId::None;
    prev_import_ir_id = import_ir_inst.ir_id();
  }
}

// Fills the previous function id, type id, and import ir id.
inline auto FillPrevEntityInfo(Context& context,
                               const SemIR::ImportIRInst& import_ir_inst,
                               SemIR::Inst decl_val,
                               SemIR::FunctionId& prev_entity_id,
                               SemIR::TypeId& prev_type_id,
                               SemIR::ImportIRId& prev_import_ir_id) -> void {
  // Verify the decl so that things like aliases are name conflicts.
  const auto* import_ir =
      context.import_irs().Get(import_ir_inst.ir_id()).sem_ir;
  if (!import_ir->insts().Is<SemIR::FunctionDecl>(import_ir_inst.inst_id())) {
    return;
  }

  if (auto struct_value = decl_val.TryAs<SemIR::StructValue>()) {
    if (auto function_type = context.types().TryGetAs<SemIR::FunctionType>(
            struct_value->type_id)) {
      prev_entity_id = function_type->function_id;
      prev_type_id = struct_value->type_id;
      prev_import_ir_id = import_ir_inst.ir_id();
    }
  }
}

// Fills the previous interface id, type id, and import ir id.
inline auto FillPrevEntityInfo(Context& context,
                               const SemIR::ImportIRInst& import_ir_inst,
                               SemIR::Inst decl_val,
                               SemIR::InterfaceId& prev_entity_id,
                               SemIR::TypeId& prev_type_id,
                               SemIR::ImportIRId& prev_import_ir_id) -> void {
  // Verify the decl so that things like aliases are name conflicts.
  const auto* import_ir =
      context.import_irs().Get(import_ir_inst.ir_id()).sem_ir;
  if (!import_ir->insts().Is<SemIR::InterfaceDecl>(import_ir_inst.inst_id())) {
    return;
  }

  if (auto facet_type = decl_val.TryAs<SemIR::FacetType>()) {
    auto declared_facet_type =
        context.declared_facet_types().Get(facet_type->declared_facet_type_id);
    prev_entity_id = declared_facet_type.extend_constraints[0].interface_id;
    prev_type_id = SemIR::TypeId::None;
    prev_import_ir_id = import_ir_inst.ir_id();
  }
}

// Fills the previous named constraint id, type id, and import ir id.
inline auto FillPrevEntityInfo(Context& context,
                               const SemIR::ImportIRInst& import_ir_inst,
                               SemIR::Inst decl_val,
                               SemIR::NamedConstraintId& prev_entity_id,
                               SemIR::TypeId& prev_type_id,
                               SemIR::ImportIRId& prev_import_ir_id) -> void {
  // Verify the decl so that things like aliases are name conflicts.
  const auto* import_ir =
      context.import_irs().Get(import_ir_inst.ir_id()).sem_ir;
  if (!import_ir->insts().Is<SemIR::NamedConstraintDecl>(
          import_ir_inst.inst_id())) {
    return;
  }

  if (auto facet_type = decl_val.TryAs<SemIR::FacetType>()) {
    auto declared_facet_type =
        context.declared_facet_types().Get(facet_type->declared_facet_type_id);
    prev_entity_id =
        declared_facet_type.extend_named_constraints[0].named_constraint_id;
    prev_type_id = SemIR::TypeId::None;
    prev_import_ir_id = import_ir_inst.ir_id();
  }
}

// Provides type traits and data for merging redeclarations.
template <typename EntityT>
struct MergeRedeclEntityInfo;

// Information for merging redeclarations of classes.
template <>
struct MergeRedeclEntityInfo<SemIR::Class> {
  using EntityIdT = SemIR::ClassId;
  using EntityT = SemIR::Class;
  using EntityDeclT = SemIR::ClassDecl;

  static constexpr auto DeclTokenKind = Lex::TokenKind::Class;

  EntityDeclT& new_entity_decl;
  const EntityT& new_entity;

  auto GetPrevEntity(Context& context, EntityIdT prev_entity_id) -> EntityT& {
    return context.classes().Get(prev_entity_id);
  }
};

// Information for merging redeclarations of functions.
template <>
struct MergeRedeclEntityInfo<SemIR::Function> {
  using EntityIdT = SemIR::FunctionId;
  using EntityT = SemIR::Function;
  using EntityDeclT = SemIR::FunctionDecl;

  static constexpr auto DeclTokenKind = Lex::TokenKind::Fn;

  EntityDeclT& new_entity_decl;
  const EntityT& new_entity;

  auto GetPrevEntity(Context& context, EntityIdT prev_entity_id) -> EntityT& {
    return context.functions().Get(prev_entity_id);
  }
};

// Information for merging redeclarations of interfaces.
template <>
struct MergeRedeclEntityInfo<SemIR::Interface> {
  using EntityIdT = SemIR::InterfaceId;
  using EntityT = SemIR::Interface;
  using EntityDeclT = SemIR::InterfaceDecl;

  static constexpr auto DeclTokenKind = Lex::TokenKind::Interface;

  EntityDeclT& new_entity_decl;
  const EntityT& new_entity;

  auto GetPrevEntity(Context& context, EntityIdT prev_entity_id) -> EntityT& {
    return context.interfaces().Get(prev_entity_id);
  }
};

// Information for merging redeclarations of named constraints.
template <>
struct MergeRedeclEntityInfo<SemIR::NamedConstraint> {
  using EntityIdT = SemIR::NamedConstraintId;
  using EntityT = SemIR::NamedConstraint;
  using EntityDeclT = SemIR::NamedConstraintDecl;

  static constexpr auto DeclTokenKind = Lex::TokenKind::Constraint;

  EntityDeclT& new_entity_decl;
  const EntityT& new_entity;

  auto GetPrevEntity(Context& context, EntityIdT prev_entity_id) -> EntityT& {
    return context.named_constraints().Get(prev_entity_id);
  }
};

// Tries to merge new_entity into prev_entity_id. Since new_entity won't have a
// definition even if one is upcoming, set is_definition to indicate the planned
// result.
//
// If merging is successful, returns the previous declaration.
// Otherwise, returns nullopt. Prints a diagnostic when appropriate.
template <typename EntityT>
auto TryMergeRedecl(Context& context,
                    const DeclNameStack::NameContext& name_context,
                    std::optional<SemIR::ScopeLookupResult> lookup_result,
                    MergeRedeclEntityInfo<EntityT> entity_info,
                    bool is_definition) -> bool {
  constexpr bool IsClass = std::is_same_v<EntityT, SemIR::Class>;
  constexpr bool IsFunction = std::is_same_v<EntityT, SemIR::Function>;
  constexpr bool IsInterface = std::is_same_v<EntityT, SemIR::Interface>;
  constexpr bool IsNamedConstraint =
      std::is_same_v<EntityT, SemIR::NamedConstraint>;

  if constexpr (IsFunction) {
    CARBON_CHECK(!lookup_result.has_value());
    // Diagnose if we are declaring a poisoned name. However, don't diagnose
    // at impl scope: if the name was referenced before being declared, we
    // will have produced an error already.
    if (name_context.state == DeclNameStack::NameContext::State::Poisoned) {
      if (!context.name_scopes().InstIs<SemIR::ImplDecl>(
              name_context.parent_scope_id)) {
        DiagnosePoisonedName(context, name_context.name_id_for_new_inst(),
                             name_context.poisoning_loc_id,
                             name_context.loc_id);
      }
      return false;
    }
  } else if constexpr (IsClass || IsInterface || IsNamedConstraint) {
    CARBON_CHECK(lookup_result.has_value());
    if (lookup_result->is_poisoned()) {
      DiagnosePoisonedName(context, name_context.name_id_for_new_inst(),
                           lookup_result->poisoning_loc_id(),
                           name_context.loc_id);
      return false;
    }

    if (!lookup_result->is_found()) {
      return false;
    }
  } else {
    CARBON_FATAL("Unhandled entity type.");
  }

  auto prev_id = lookup_result ? lookup_result->target_inst_id()
                               : name_context.prev_inst_id();
  if (!prev_id.has_value()) {
    return false;
  }
  auto prev = context.insts().Get(prev_id);

  auto prev_entity_id = MergeRedeclEntityInfo<EntityT>::EntityIdT::None;
  auto prev_type_id = SemIR::TypeId::None;
  auto prev_import_ir_id = SemIR::ImportIRId::None;
  CARBON_KIND_SWITCH(prev) {
    case CARBON_KIND(SemIR::AssociatedEntity assoc_entity): {
      if constexpr (IsFunction) {
        // This is a function in an interface definition scope.
        auto function_decl =
            context.insts().GetAs<SemIR::FunctionDecl>(assoc_entity.decl_id);
        prev_entity_id = function_decl.function_id;
        prev_type_id = function_decl.type_id;
      }
      break;
    }
    case CARBON_KIND(SemIR::ClassDecl class_decl): {
      if constexpr (IsClass) {
        prev_entity_id = class_decl.class_id;
      }
      break;
    }
    case CARBON_KIND(SemIR::FunctionDecl function_decl): {
      if constexpr (IsFunction) {
        prev_entity_id = function_decl.function_id;
        prev_type_id = function_decl.type_id;
      }
      break;
    }
    case CARBON_KIND(SemIR::InterfaceDecl interface_decl): {
      if constexpr (IsInterface) {
        prev_entity_id = interface_decl.interface_id;
      }
      break;
    }
    case CARBON_KIND(SemIR::NamedConstraintDecl named_constraint_decl): {
      if constexpr (IsNamedConstraint) {
        prev_entity_id = named_constraint_decl.named_constraint_id;
      }
      break;
    }
    case CARBON_KIND(SemIR::ImportRefLoaded import_ref): {
      // TODO: Should we get canonical inst for all entity types?
      auto import_ir_inst = [&]() -> SemIR::ImportIRInst {
        if constexpr (IsClass || IsInterface || IsNamedConstraint) {
          return context.import_ir_insts().Get(import_ref.import_ir_inst_id);
        } else if constexpr (IsFunction) {
          return GetCanonicalImportIRInst(context, prev_id);
        } else {
          CARBON_FATAL("Unhandled entity type.");
        }
      }();
      auto decl_val = context.insts().Get(
          context.constant_values().GetConstantInstId(prev_id));
      FillPrevEntityInfo(context, import_ir_inst, decl_val, prev_entity_id,
                         prev_type_id, prev_import_ir_id);
      break;
    }
    default: {
      break;
    }
  }

  if (!prev_entity_id.has_value()) {
    // This is a redeclaration with a different entity kind.
    DiagnoseDuplicateName(context, name_context.name_id, name_context.loc_id,
                          SemIR::LocId(prev_id));
    return false;
  }

  auto& prev_entity = entity_info.GetPrevEntity(context, prev_entity_id);

  if constexpr (IsClass || IsInterface || IsNamedConstraint) {
    if (!CheckRedeclParamsMatch(context, DeclParams(entity_info.new_entity),
                                DeclParams(prev_entity))) {
      // Mismatch is diagnosed already if found.
      return false;
    }
  } else if constexpr (IsFunction) {
    if (!CheckFunctionTypeMatches(context, entity_info.new_entity,
                                  prev_entity)) {
      // Mismatch is diagnosed already if found.
      return false;
    }
  } else {
    CARBON_FATAL("Unhandled entity type.");
  }

  DiagnoseIfInvalidRedecl(
      context, MergeRedeclEntityInfo<EntityT>::DeclTokenKind,
      prev_entity.name_id,
      RedeclInfo(entity_info.new_entity,
                 SemIR::LocId(entity_info.new_entity.latest_decl_id()),
                 is_definition),
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
    return false;
  }

  if (!prev_entity.first_owning_decl_id.has_value()) {
    prev_entity.first_owning_decl_id =
        entity_info.new_entity.first_owning_decl_id;
  }

  if (is_definition) {
    prev_entity.MergeDefinition(entity_info.new_entity);
  }

  auto replace_prev_inst = prev_import_ir_id.has_value();
  if constexpr (IsClass) {
    replace_prev_inst |=
        prev_entity.is_extern && !entity_info.new_entity.is_extern;
  }
  if (replace_prev_inst) {
    ReplacePrevInstForMerge(context, entity_info.new_entity.parent_scope_id,
                            prev_entity.name_id,
                            entity_info.new_entity.first_owning_decl_id);
  }

  // When merging, use the existing entity rather than adding a new one.
  if constexpr (IsClass) {
    // TODO: Fix `extern` logic. It doesn't work correctly, but doesn't seem
    // worth ripping out because existing code may incrementally help.
    entity_info.new_entity_decl.class_id = prev_entity_id;
    entity_info.new_entity_decl.type_id = prev.type_id();
    // TODO: Validate that the redeclaration doesn't set an access modifier.
  } else if constexpr (IsFunction) {
    entity_info.new_entity_decl.function_id = prev_entity_id;
    entity_info.new_entity_decl.type_id = prev_type_id;
  } else if constexpr (IsInterface) {
    entity_info.new_entity_decl.interface_id = prev_entity_id;
    entity_info.new_entity_decl.type_id = prev.type_id();
  } else if constexpr (IsNamedConstraint) {
    entity_info.new_entity_decl.named_constraint_id = prev_entity_id;
    entity_info.new_entity_decl.type_id = prev.type_id();
  } else {
    CARBON_FATAL("Unhandled entity type.");
  }

  return true;
}

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_MERGE_H_
