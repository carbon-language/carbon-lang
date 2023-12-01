// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/modifiers_allowed.h"

namespace Carbon::Check {

static auto ReportNotAllowed(Context& context, Parse::Node modifier_node,
                             llvm::StringRef decl_name,
                             llvm::StringRef context_string,
                             Parse::Node context_node) -> void {
  CARBON_DIAGNOSTIC(ModifierNotAllowedOn, Error, "`{0}` not allowed on {1}{2}.",
                    std::string, std::string, std::string);
  auto diag = context.emitter().Build(modifier_node, ModifierNotAllowedOn,
                                      context.TextForNode(modifier_node),
                                      decl_name.str(), context_string.str());
  if (context_node.is_valid()) {
    CARBON_DIAGNOSTIC(ModifierNotInContext, Note,
                      "Containing definition here.");
    diag.Note(context_node, ModifierNotInContext);
  }
  diag.Emit();
}

auto LimitModifiersOnDecl(Context& context, KeywordModifierSet allowed,
                          llvm::StringRef decl_name, Parse::Node context_node)
    -> KeywordModifierSet {
  auto& s = context.decl_state_stack().innermost();
  auto not_allowed = s.modifier_set & ~allowed;
  if (!!(not_allowed & KeywordModifierSet::Access)) {
    ReportNotAllowed(context, s.saw_access_modifier, decl_name, "",
                     context_node);
    not_allowed = not_allowed & ~KeywordModifierSet::Access;
    s.saw_access_modifier = Parse::Node::Invalid;
  }
  if (!!not_allowed) {
    ReportNotAllowed(context, s.saw_decl_modifier, decl_name, "", context_node);
    s.saw_decl_modifier = Parse::Node::Invalid;
  }
  s.modifier_set &= allowed;

  return s.modifier_set;
}

auto ForbidModifiersOnDecl(Context& context, KeywordModifierSet forbidden,
                           llvm::StringRef decl_name,
                           llvm::StringRef context_string,
                           Parse::Node context_node) -> void {
  auto& s = context.decl_state_stack().innermost();
  auto not_allowed = s.modifier_set & forbidden;
  if (!!(not_allowed & KeywordModifierSet::Access)) {
    ReportNotAllowed(context, s.saw_access_modifier, decl_name, context_string,
                     context_node);
    not_allowed = not_allowed & ~KeywordModifierSet::Access;
    s.saw_access_modifier = Parse::Node::Invalid;
  }
  if (!!not_allowed) {
    ReportNotAllowed(context, s.saw_decl_modifier, decl_name, context_string,
                     context_node);
    s.saw_decl_modifier = Parse::Node::Invalid;
  }
  s.modifier_set = s.modifier_set & ~forbidden;
}

auto CheckAccessModifiersOnDecl(Context& context, llvm::StringRef decl_name)
    -> void {
  switch (context.decl_state_stack().containing().kind) {
    case DeclState::FileScope:
      ForbidModifiersOnDecl(
          context, KeywordModifierSet::Protected, decl_name,
          " at file scope, `protected` is only allowed on class members");
      break;

    case DeclState::Class:
      // Both `private` and `protected` allowed in a class definition.
      break;

    default:
      // Otherwise neither `private` nor `protected` allowed.
      ForbidModifiersOnDecl(context, KeywordModifierSet::Protected, decl_name,
                            ", `protected` is only allowed on class members");
      ForbidModifiersOnDecl(
          context, KeywordModifierSet::Private, decl_name,
          ", `private` is only allowed on class members and at file scope");
      break;
  }
}

auto RequireDefaultFinalOnlyInInterfaces(Context& context,
                                         llvm::StringRef decl_name)
    -> KeywordModifierSet {
  auto& s = context.decl_state_stack().innermost();
  if (context.decl_state_stack().containing().kind != DeclState::Interface) {
    ForbidModifiersOnDecl(context, KeywordModifierSet::Interface, decl_name,
                          " outside of an interface");
  }

  return s.modifier_set;
}

}  // namespace Carbon::Check
