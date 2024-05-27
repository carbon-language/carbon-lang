// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_MERGE_H_
#define CARBON_TOOLCHAIN_CHECK_MERGE_H_

#include "toolchain/check/context.h"
#include "toolchain/check/subst.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Information on new and previous declarations for CheckIsAllowedRedecl.
struct RedeclInfo {
  // The associated diagnostic location.
  SemIRLoc loc;
  // True if a definition.
  bool is_definition;
  // True if an `extern` declaration.
  bool is_extern;
};

// Checks if a redeclaration is allowed prior to merging. This may emit a
// diagnostic, but diagnostics do not prevent merging.
//
// The kinds of things this verifies are:
// - A declaration is not redundant.
// - A definition doesn't redefine a prior definition.
// - The use of `extern` is consistent within a library.
// - Multiple libraries do not declare non-`extern`.
auto CheckIsAllowedRedecl(Context& context, Lex::TokenKind decl_kind,
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
  template <typename Entity>
  explicit DeclParams(const Entity& entity)
      : decl_id(entity.decl_id),
        implicit_param_refs_id(entity.implicit_param_refs_id),
        param_refs_id(entity.param_refs_id) {}

  // The declaration of the entity.
  SemIR::InstId decl_id;
  // The implicit parameters of the entity. Can be Invalid if there is no
  // implicit parameter list.
  SemIR::InstBlockId implicit_param_refs_id;
  // The explicit parameters of the entity. Can be Invalid if there is no
  // explicit parameter list.
  SemIR::InstBlockId param_refs_id;
};

// Checks that the parameters in a redeclaration of an entity match the
// parameters in the prior declaration. If not, produces a diagnostic and
// returns false.
auto CheckRedeclParamsMatch(Context& context, const DeclParams& new_entity,
                            const DeclParams& prev_entity,
                            Substitutions substitutions) -> bool;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_MERGE_H_
