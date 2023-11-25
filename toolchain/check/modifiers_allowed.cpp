// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/modifiers_allowed.h"

namespace Carbon::Check {

auto ModifiersAllowedOnDecl(Context& context, KeywordModifierSet allowed,
                            llvm::StringRef decl_name)
    -> std::pair<KeywordModifierSet, Parse::Node> {
  auto& s = context.innermost_decl();
  auto not_allowed = s.found.GetRaw() & ~allowed.GetRaw();
  CARBON_DIAGNOSTIC(ModifierNotAllowedOn, Error, "`{0}` not allowed on {1}.",
                    llvm::StringRef, llvm::StringRef);
  if (not_allowed & KeywordModifierSet::Access) {
    context.emitter().Emit(s.saw_access_mod, ModifierNotAllowedOn,
                           context.TextForNode(s.saw_access_mod), decl_name);
    not_allowed &= ~KeywordModifierSet::Access;
    s.saw_access_mod = Parse::Node::Invalid;
  }
  if (not_allowed) {
    context.emitter().Emit(s.saw_decl_mod, ModifierNotAllowedOn,
                           context.TextForNode(s.saw_decl_mod), decl_name);
    s.saw_decl_mod = Parse::Node::Invalid;
  }
  s.found = KeywordModifierSet::RawEnum(s.found.GetRaw() & allowed.GetRaw());

  return {s.found, s.first_node};
}

}  // namespace Carbon::Check
