// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/modifiers_allowed.h"

namespace Carbon::Check {

static auto ReportNotAllowed(Context& context, Parse::Node modifier_node,
                             llvm::StringRef decl_name,
                             llvm::StringRef context_string = "",
                             Parse::Node context_node = Parse::Node::Invalid) {
  CARBON_DIAGNOSTIC(ModifierNotAllowedOn, Error, "`{0}` not allowed on {1}{2}.",
                    llvm::StringRef, llvm::StringRef, llvm::StringRef);
  if (context_node == Parse::Node::Invalid) {
    context.emitter().Emit(modifier_node, ModifierNotAllowedOn,
                           context.TextForNode(modifier_node), decl_name,
                           context_string);
  } else {
    CARBON_DIAGNOSTIC(ModifierNotInContext, Note,
                      "Containing definition here.");
    context.emitter()
        .Build(modifier_node, ModifierNotAllowedOn,
               context.TextForNode(modifier_node), decl_name, context_string)
        .Note(context_node, ModifierNotInContext)
        .Emit();
  }
}

static auto ReportNotAllowedInContext(Context& context,
                                      Parse::Node modifier_node,
                                      llvm::StringRef decl_name) {
  switch (context.containing_decl().kind) {
    case DeclState::FileScope:
      ReportNotAllowed(context, modifier_node, decl_name, " at file scope");
      break;

    case DeclState::Class:
      ReportNotAllowed(context, modifier_node, decl_name,
                       " in a `class` definition",
                       context.containing_decl().first_node);
      break;

      // Otherwise neither `private` nor `protected` allowed.
    case DeclState::NamedConstraint:
      ReportNotAllowed(context, modifier_node, decl_name,
                       " in a `constraint` definition",
                       context.containing_decl().first_node);
      break;

    case DeclState::Fn:
      ReportNotAllowed(context, modifier_node, decl_name,
                       " in a `fn` definition",
                       context.containing_decl().first_node);
      break;

    case DeclState::Interface:
      ReportNotAllowed(context, modifier_node, decl_name,
                       " in an `interface` definition",
                       context.containing_decl().first_node);
      break;

    case DeclState::Let:
      ReportNotAllowed(context, modifier_node, decl_name,
                       " in a `let` definition",
                       context.containing_decl().first_node);
      break;

    case DeclState::Var:
      ReportNotAllowed(context, modifier_node, decl_name,
                       " in a `var` definition",
                       context.containing_decl().first_node);
      break;
  }
}

static auto NoAccessControlAllowed(Context& context, llvm::StringRef decl_name)
    -> void {
  auto& s = context.innermost_decl();
  if (s.found.Has(KeywordModifierSet::Access)) {
    ReportNotAllowedInContext(context, s.saw_access_mod, decl_name);
    s.found = s.found.Clear(KeywordModifierSet::Access);
  }
}

auto CheckAccessModifiersOnDecl(Context& context, llvm::StringRef decl_name)
    -> void {
  switch (context.containing_decl().kind) {
    case DeclState::FileScope:
      // Only `private` allowed at file scope.
      if (context.innermost_decl().found.HasProtected()) {
        NoAccessControlAllowed(context, decl_name);
      }
      break;

    case DeclState::Class:
      // Both `private` and `protected` allowed in a class definition.
      break;

    default:
      // Otherwise neither `private` nor `protected` allowed.
      NoAccessControlAllowed(context, decl_name);
      break;
  }
}

auto ModifiersAllowedOnDecl(Context& context, KeywordModifierSet allowed,
                            llvm::StringRef decl_name,
                            ShouldReportContext report_context)
    -> KeywordModifierSet {
  auto& s = context.innermost_decl();
  // If non-access modifier not in `allowed`
  if (s.found.Has(~allowed.GetRaw() & ~KeywordModifierSet::Access)) {
    if (report_context == IncludeContext) {
      ReportNotAllowedInContext(context, s.saw_decl_mod, decl_name);
    } else {
      ReportNotAllowed(context, s.saw_decl_mod, decl_name);
    }
    s.saw_decl_mod = Parse::Node::Invalid;
  }
  s.found = s.found.Clear(~(allowed.GetRaw() | KeywordModifierSet::Access));
  return s.found;
}

auto ModifiersAllowedOnDeclCustomContext(Context& context,
                                         KeywordModifierSet allowed,
                                         llvm::StringRef decl_name,
                                         llvm::StringRef context_string)
    -> KeywordModifierSet {
  auto& s = context.innermost_decl();
  // If non-access modifier not in `allowed`
  if (s.found.Has(~allowed.GetRaw() & ~KeywordModifierSet::Access)) {
    ReportNotAllowed(context, s.saw_decl_mod, decl_name, context_string,
                     context.containing_decl().first_node);
    s.saw_decl_mod = Parse::Node::Invalid;
  }
  s.found = s.found.Clear(~(allowed.GetRaw() | KeywordModifierSet::Access));
  return s.found;
}

}  // namespace Carbon::Check
