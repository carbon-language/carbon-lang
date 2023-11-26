// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/modifiers_allowed.h"

namespace Carbon::Check {

static auto ReportNotAllowed(Context& context, Parse::Node modifier_node,
                             llvm::StringRef decl_name,
                             Parse::Node context_node) {
  CARBON_DIAGNOSTIC(ModifierNotAllowedOn, Error, "`{0}` not allowed on {1}.",
                    llvm::StringRef, llvm::StringRef);
  if (context_node == Parse::Node::Invalid) {
    context.emitter().Emit(modifier_node, ModifierNotAllowedOn,
                           context.TextForNode(modifier_node), decl_name);
  } else {
    CARBON_DIAGNOSTIC(ModifierNotInContext, Note,
                      "Containing definition here.");
    context.emitter()
        .Build(modifier_node, ModifierNotAllowedOn,
               context.TextForNode(modifier_node), decl_name)
        .Note(context_node, ModifierNotInContext)
        .Emit();
  }
}

auto ModifiersAllowedOnDecl(Context& context, KeywordModifierSet allowed,
                            llvm::StringRef decl_name, Parse::Node context_node)
    -> KeywordModifierSet {
  auto& s = context.innermost_decl();
  auto not_allowed = s.found.GetRaw() & ~allowed.GetRaw();
  if (not_allowed & KeywordModifierSet::Access) {
    ReportNotAllowed(context, s.saw_access_mod, decl_name, context_node);
    not_allowed &= ~KeywordModifierSet::Access;
    s.saw_access_mod = Parse::Node::Invalid;
  }
  if (not_allowed) {
    ReportNotAllowed(context, s.saw_decl_mod, decl_name, context_node);
    s.saw_decl_mod = Parse::Node::Invalid;
  }
  s.found = KeywordModifierSet::RawEnum(s.found.GetRaw() & allowed.GetRaw());

  return s.found;
}

}  // namespace Carbon::Check
