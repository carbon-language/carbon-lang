// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/decl_state.h"
#include "toolchain/lex/token_kind.h"

namespace Carbon::Check {

CARBON_DIAGNOSTIC(ModifierPrevious, Note, "`{0}` previously appeared here.",
                  Lex::TokenKind);

static auto EmitRepeatedDiagnostic(Context& context, Parse::NodeId first_node,
                                   Parse::NodeId second_node) -> void {
  CARBON_DIAGNOSTIC(ModifierRepeated, Error, "`{0}` repeated on declaration.",
                    Lex::TokenKind);
  context.emitter()
      .Build(second_node, ModifierRepeated, context.token_kind(second_node))
      .Note(first_node, ModifierPrevious, context.token_kind(first_node))
      .Emit();
}

static auto EmitNotAllowedWithDiagnostic(Context& context,
                                         Parse::NodeId first_node,
                                         Parse::NodeId second_node) -> void {
  CARBON_DIAGNOSTIC(ModifierNotAllowedWith, Error,
                    "`{0}` not allowed on declaration with `{1}`.",
                    Lex::TokenKind, Lex::TokenKind);
  context.emitter()
      .Build(second_node, ModifierNotAllowedWith,
             context.token_kind(second_node), context.token_kind(first_node))
      .Note(first_node, ModifierPrevious, context.token_kind(first_node))
      .Emit();
}

static auto HandleModifier(Context& context, Parse::NodeId parse_node,
                           KeywordModifierSet keyword) -> bool {
  auto& s = context.decl_state_stack().innermost();
  bool is_access = !!(keyword & KeywordModifierSet::Access);
  auto& saw_modifier = is_access ? s.saw_access_modifier : s.saw_decl_modifier;
  if (!!(s.modifier_set & keyword)) {
    EmitRepeatedDiagnostic(context, saw_modifier, parse_node);
  } else if (saw_modifier.is_valid()) {
    EmitNotAllowedWithDiagnostic(context, saw_modifier, parse_node);
  } else if (is_access && s.saw_decl_modifier.is_valid()) {
    CARBON_DIAGNOSTIC(ModifierMustAppearBefore, Error,
                      "`{0}` must appear before `{1}`.", Lex::TokenKind,
                      Lex::TokenKind);
    context.emitter()
        .Build(parse_node, ModifierMustAppearBefore,
               context.token_kind(parse_node),
               context.token_kind(s.saw_decl_modifier))
        .Note(s.saw_decl_modifier, ModifierPrevious,
              context.token_kind(s.saw_decl_modifier))
        .Emit();
  } else {
    s.modifier_set |= keyword;
    saw_modifier = parse_node;
  }
  return true;
}

#define CARBON_PARSE_NODE_KIND(...)
#define CARBON_PARSE_NODE_KIND_TOKEN_MODIFIER(Name, ...)                    \
  auto Handle##Name##Modifier(Context& context,                             \
                              Parse::Name##ModifierId parse_node) -> bool { \
    return HandleModifier(context, parse_node, KeywordModifierSet::Name);   \
  }
#include "toolchain/parse/node_kind.def"

}  // namespace Carbon::Check
