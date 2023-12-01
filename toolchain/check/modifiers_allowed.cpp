// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/modifiers_allowed.h"

namespace Carbon::Check {

static auto ReportNotAllowed(Context& context, Parse::Node modifier_node,
                             llvm::StringRef decl_name,
                             llvm::StringRef context_string,
                             Parse::Node context_node) {
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

auto ModifiersAllowedOnDecl(Context& context, KeywordModifierSet allowed,
                            llvm::StringRef decl_name, Parse::Node context_node)
    -> KeywordModifierSet {
  auto& s = context.decl_state_stack().innermost();
  auto not_allowed = s.found & ~allowed;
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
  s.found &= allowed;

  return s.found;
}

auto ForbidModifiers(Context& context, KeywordModifierSet forbidden,
                     llvm::StringRef decl_name, llvm::StringRef context_string,
                     Parse::Node context_node) -> void {
  auto& s = context.decl_state_stack().innermost();
  auto not_allowed = s.found & forbidden;
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
  s.found = s.found & ~forbidden;
}

auto CheckAccessModifiersOnDecl(Context& context, llvm::StringRef decl_name)
    -> void {
  switch (context.decl_state_stack().containing().kind) {
    case DeclState::FileScope:
      ForbidModifiers(
          context, KeywordModifierSet::Protected, decl_name,
          " at file scope, `protected` is only allowed on class members");
      break;

    case DeclState::Class:
      // Both `private` and `protected` allowed in a class definition.
      break;

    default:
      // Otherwise neither `private` nor `protected` allowed.
      ForbidModifiers(context, KeywordModifierSet::Protected, decl_name,
                      ", `protected` is only allowed on class members");
      ForbidModifiers(
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
    ForbidModifiers(context, KeywordModifierSet::Interface, decl_name,
                    " outside of an interface",
                    context.decl_state_stack().containing().first_node);
  }

  return s.found;
}

}  // namespace Carbon::Check
