// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/lex/token_kind.h"

namespace Carbon::Check {

CARBON_DIAGNOSTIC(ModifierPrevious, Note, "`{0}` previously appeared here.",
                  std::string);

static auto EmitRepeatedDiagnostic(Context& context, Parse::Node first_node,
                                   Parse::Node second_node) -> void {
  CARBON_DIAGNOSTIC(ModifierRepeated, Error, "`{0}` repeated on declaration.",
                    std::string);
  context.emitter()
      .Build(second_node, ModifierRepeated, context.TextForNode(second_node))
      .Note(first_node, ModifierPrevious, context.TextForNode(first_node))
      .Emit();
}

static auto EmitNotAllowedWithDiagnostic(Context& context,
                                         Parse::Node first_node,
                                         Parse::Node second_node) -> void {
  CARBON_DIAGNOSTIC(ModifierNotAllowedWith, Error,
                    "`{0}` not allowed on declaration with `{1}`.", std::string,
                    std::string);
  context.emitter()
      .Build(second_node, ModifierNotAllowedWith,
             context.TextForNode(second_node), context.TextForNode(first_node))
      .Note(first_node, ModifierPrevious, context.TextForNode(first_node))
      .Emit();
}

static auto GetAccessModifierEnum(Lex::TokenKind token_kind)
    -> KeywordModifierSet {
  switch (token_kind) {
    case Lex::TokenKind::Private:
      return KeywordModifierSet::Private;
    case Lex::TokenKind::Protected:
      return KeywordModifierSet::Protected;
    default:
      CARBON_FATAL() << "Unhandled access modifier keyword";
  }
}

auto HandleAccessModifierKeyword(Context& context, Parse::Node parse_node)
    -> bool {
  auto keyword = GetAccessModifierEnum(
      context.tokens().GetKind(context.parse_tree().node_token(parse_node)));
  auto& s = context.decl_state_stack().innermost();
  if (!!(s.modifier_set & keyword)) {
    EmitRepeatedDiagnostic(context, s.saw_access_modifier, parse_node);
  } else if (s.saw_access_modifier.is_valid()) {
    EmitNotAllowedWithDiagnostic(context, s.saw_access_modifier, parse_node);
  } else if (s.saw_decl_modifier.is_valid()) {
    CARBON_DIAGNOSTIC(ModifierMustAppearBefore, Error,
                      "`{0}` must appear before `{1}`.", std::string,
                      std::string);
    context.emitter()
        .Build(parse_node, ModifierMustAppearBefore,
               context.TextForNode(parse_node),
               context.TextForNode(s.saw_decl_modifier))
        .Note(s.saw_decl_modifier, ModifierPrevious,
              context.TextForNode(s.saw_decl_modifier))
        .Emit();
  } else {
    s.modifier_set |= keyword;
    s.saw_access_modifier = parse_node;
    s.first_node = parse_node;
  }

  return true;
}

static auto GetDeclModifierEnum(Lex::TokenKind token_kind)
    -> KeywordModifierSet {
  switch (token_kind) {
    case Lex::TokenKind::Abstract:
      return KeywordModifierSet::Abstract;
    case Lex::TokenKind::Base:
      return KeywordModifierSet::Base;
    case Lex::TokenKind::Default:
      return KeywordModifierSet::Default;
    case Lex::TokenKind::Final:
      return KeywordModifierSet::Final;
    case Lex::TokenKind::Impl:
      return KeywordModifierSet::Impl;
    case Lex::TokenKind::Virtual:
      return KeywordModifierSet::Virtual;
    default:
      CARBON_FATAL() << "Unhandled declaration modifier keyword";
  }
}

auto HandleDeclModifierKeyword(Context& context, Parse::Node parse_node)
    -> bool {
  auto keyword = GetDeclModifierEnum(
      context.tokens().GetKind(context.parse_tree().node_token(parse_node)));
  auto& s = context.decl_state_stack().innermost();
  if (!!(s.modifier_set & keyword)) {
    EmitRepeatedDiagnostic(context, s.saw_decl_modifier, parse_node);
  } else if (s.saw_decl_modifier.is_valid()) {
    EmitNotAllowedWithDiagnostic(context, s.saw_decl_modifier, parse_node);
  } else {
    s.modifier_set |= keyword;
    s.saw_decl_modifier = parse_node;
    if (s.saw_access_modifier == Parse::Node::Invalid) {
      s.first_node = parse_node;
    }
  }
  return true;
}

}  // namespace Carbon::Check
