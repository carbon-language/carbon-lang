// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_MODIFIERS_H_
#define CARBON_TOOLCHAIN_CHECK_MODIFIERS_H_

#include "toolchain/check/context.h"

namespace Carbon::Check {

// Reports a diagnostic if access control modifiers on this are not allowed for
// a declaration in `parent_scope_inst`, and updates the declaration state in
// `context`.
//
// `parent_scope_inst` may be nullopt for a declaration in a block scope.
auto CheckAccessModifiersOnDecl(Context& context, Lex::TokenKind decl_kind,
                                std::optional<SemIR::Inst> parent_scope_inst)
    -> void;

// Reports a diagnostic if the method function modifiers `abstract`, `virtual`,
// or `impl` are present but not permitted on a function declaration in
// `parent_scope_inst`.
//
// `parent_scope_inst` may be nullopt for a declaration in a block scope.
auto CheckMethodModifiersOnFunction(
    Context& context, SemIR::InstId parent_scope_inst_id,
    std::optional<SemIR::Inst> parent_scope_inst) -> void;

// Like `LimitModifiersOnDecl`, except says which modifiers are forbidden, and a
// `context_string` (and optional `context_loc_id`) specifying the context in
// which those modifiers are forbidden.
// TODO: Take another look at diagnostic phrasing for callers.
auto ForbidModifiersOnDecl(Context& context, KeywordModifierSet forbidden,
                           Lex::TokenKind decl_kind,
                           llvm::StringRef context_string,
                           SemIR::LocId context_loc_id = SemIR::LocId::Invalid)
    -> void;

// Reports a diagnostic (using `decl_kind`) if modifiers on this declaration are
// not in `allowed`. Updates the declaration state in
// `context.decl_state_stack()`.
inline auto LimitModifiersOnDecl(Context& context, KeywordModifierSet allowed,
                                 Lex::TokenKind decl_kind) -> void {
  ForbidModifiersOnDecl(context, ~allowed, decl_kind, "");
}

// Restricts the `extern` modifier to only be used on namespace-scoped
// declarations, diagnosing and removing it on:
// - `extern` on a definition.
// - `extern` on a scoped entity.
//
// `parent_scope_inst` may be nullopt for a declaration in a block scope.
auto RestrictExternModifierOnDecl(Context& context, Lex::TokenKind decl_kind,
                                  std::optional<SemIR::Inst> parent_scope_inst,
                                  bool is_definition) -> void;

// Report a diagonostic if `default` and `final` modifiers are used on
// declarations where they are not allowed. Right now they are only allowed
// inside interfaces.
//
// `parent_scope_inst` may be nullopt for a declaration in a block scope.
auto RequireDefaultFinalOnlyInInterfaces(
    Context& context, Lex::TokenKind decl_kind,
    std::optional<SemIR::Inst> parent_scope_inst) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_MODIFIERS_H_
