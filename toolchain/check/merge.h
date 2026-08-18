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
};

// Updates the default values in `prev_function` to include any of those not
// previously specified and that are now specified in `new_function`.
auto MergeFunctionParamDefaultValues(Context& context,
                                     SemIR::Function& prev_function,
                                     const SemIR::Function& new_function)
    -> void;

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
                    bool is_definition) -> bool;

extern template auto TryMergeRedecl(Context&, const DeclNameStack::NameContext&,
                                    std::optional<SemIR::ScopeLookupResult>,
                                    MergeRedeclEntityInfo<SemIR::Class>, bool)
    -> bool;
extern template auto TryMergeRedecl(Context&, const DeclNameStack::NameContext&,
                                    std::optional<SemIR::ScopeLookupResult>,
                                    MergeRedeclEntityInfo<SemIR::Function>,
                                    bool) -> bool;
extern template auto TryMergeRedecl(Context&, const DeclNameStack::NameContext&,
                                    std::optional<SemIR::ScopeLookupResult>,
                                    MergeRedeclEntityInfo<SemIR::Interface>,
                                    bool) -> bool;
extern template auto TryMergeRedecl(
    Context&, const DeclNameStack::NameContext&,
    std::optional<SemIR::ScopeLookupResult>,
    MergeRedeclEntityInfo<SemIR::NamedConstraint>, bool) -> bool;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_MERGE_H_
