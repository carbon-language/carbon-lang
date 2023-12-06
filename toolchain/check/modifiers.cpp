// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/modifiers.h"

namespace Carbon::Check {

static auto ReportNotAllowed(Context& context, Parse::NodeId modifier_node,
                             Lex::TokenKind decl_kind,
                             llvm::StringRef context_string,
                             Parse::NodeId context_node) -> void {
  CARBON_DIAGNOSTIC(ModifierNotAllowedOn, Error,
                    "`{0}` not allowed on `{1}` declaration{2}.",
                    Lex::TokenKind, Lex::TokenKind, std::string);
  auto diag = context.emitter().Build(modifier_node, ModifierNotAllowedOn,
                                      context.token_kind(modifier_node),
                                      decl_kind, context_string.str());
  if (context_node.is_valid()) {
    CARBON_DIAGNOSTIC(ModifierNotInContext, Note,
                      "Containing definition here.");
    diag.Note(context_node, ModifierNotInContext);
  }
  diag.Emit();
}

auto LimitModifiersOnDecl(Context& context, KeywordModifierSet allowed,
                          Lex::TokenKind decl_kind) -> void {
  auto& s = context.decl_state_stack().innermost();
  auto not_allowed = s.modifier_set & ~allowed;
  if (!!(not_allowed & KeywordModifierSet::Access)) {
    ReportNotAllowed(context, s.saw_access_modifier, decl_kind, "",
                     Parse::NodeId::Invalid);
    not_allowed = not_allowed & ~KeywordModifierSet::Access;
    s.saw_access_modifier = Parse::NodeId::Invalid;
  }
  if (!!not_allowed) {
    ReportNotAllowed(context, s.saw_decl_modifier, decl_kind, "",
                     Parse::NodeId::Invalid);
    s.saw_decl_modifier = Parse::NodeId::Invalid;
  }
  s.modifier_set &= allowed;
}

auto ForbidModifiersOnDecl(Context& context, KeywordModifierSet forbidden,
                           Lex::TokenKind decl_kind,
                           llvm::StringRef context_string,
                           Parse::NodeId context_node) -> void {
  auto& s = context.decl_state_stack().innermost();
  auto not_allowed = s.modifier_set & forbidden;
  if (!!(not_allowed & KeywordModifierSet::Access)) {
    ReportNotAllowed(context, s.saw_access_modifier, decl_kind, context_string,
                     context_node);
    not_allowed = not_allowed & ~KeywordModifierSet::Access;
    s.saw_access_modifier = Parse::NodeId::Invalid;
  }
  if (!!not_allowed) {
    ReportNotAllowed(context, s.saw_decl_modifier, decl_kind, context_string,
                     context_node);
    s.saw_decl_modifier = Parse::NodeId::Invalid;
  }
  s.modifier_set = s.modifier_set & ~forbidden;
}

auto CheckAccessModifiersOnDecl(Context& context, Lex::TokenKind decl_kind)
    -> void {
  if (context.at_file_scope()) {
    ForbidModifiersOnDecl(
        context, KeywordModifierSet::Protected, decl_kind,
        " at file scope, `protected` is only allowed on class members");
    return;
  }

  if (auto kind = context.current_scope_kind()) {
    if (*kind == SemIR::ClassDecl::Kind) {
      // Both `private` and `protected` allowed in a class definition.
      return;
    }
  }

  // Otherwise neither `private` nor `protected` allowed.
  ForbidModifiersOnDecl(context, KeywordModifierSet::Protected, decl_kind,
                        ", `protected` is only allowed on class members");
  ForbidModifiersOnDecl(
      context, KeywordModifierSet::Private, decl_kind,
      ", `private` is only allowed on class members and at file scope");
}

auto RequireDefaultFinalOnlyInInterfaces(Context& context,
                                         Lex::TokenKind decl_kind) -> void {
  // FIXME: Skip this if *context.current_scope_kind() == SemIR::InterfaceDecl
  ForbidModifiersOnDecl(context, KeywordModifierSet::Interface, decl_kind,
                        " outside of an interface");
}

}  // namespace Carbon::Check
