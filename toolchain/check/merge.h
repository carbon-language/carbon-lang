// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_MERGE_H_
#define CARBON_TOOLCHAIN_CHECK_MERGE_H_

#include "toolchain/check/context.h"
#include "toolchain/check/subst.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Diagnoses an `extern` declaration that was not preceded by a declaration in
// the API file.
auto DiagnoseExternRequiresDeclInApiFile(Context& context, SemIRLoc loc)
    -> void;

// Information on new and previous declarations for DiagnoseIfInvalidRedecl.
struct RedeclInfo {
  explicit RedeclInfo(SemIR::EntityWithParamsBase params, SemIRLoc loc,
                      bool is_definition)
      : loc(loc),
        is_definition(is_definition),
        is_extern(params.is_extern),
        extern_library_id(params.extern_library_id) {}

  // The associated diagnostic location.
  SemIRLoc loc;
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
      : loc(base.latest_decl_id()),
        first_param_node_id(base.first_param_node_id),
        last_param_node_id(base.last_param_node_id),
        implicit_param_patterns_id(base.implicit_param_patterns_id),
        param_patterns_id(base.param_patterns_id) {}

  DeclParams(SemIRLoc loc, Parse::NodeId first_param_node_id,
             Parse::NodeId last_param_node_id,
             SemIR::InstBlockId implicit_param_patterns_id,
             SemIR::InstBlockId param_patterns_id)
      : loc(loc),
        first_param_node_id(first_param_node_id),
        last_param_node_id(last_param_node_id),
        implicit_param_patterns_id(implicit_param_patterns_id),
        param_patterns_id(param_patterns_id) {}

  // The location of the declaration of the entity.
  SemIRLoc loc;

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
auto CheckRedeclParamsMatch(
    Context& context, const DeclParams& new_entity,
    const DeclParams& prev_entity,
    SemIR::SpecificId prev_specific_id = SemIR::SpecificId::None,
    bool check_syntax = true, bool diagnose = true) -> bool;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_MERGE_H_
