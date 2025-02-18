// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/decl_introducer_state.h"
#include "toolchain/check/handle.h"
#include "toolchain/lex/token_kind.h"

namespace Carbon::Check {

CARBON_DIAGNOSTIC(ModifierPrevious, Note, "`{0}` previously appeared here",
                  Lex::TokenKind);

static auto DiagnoseRepeated(Context& context, Parse::NodeId first_node,
                             Parse::NodeId second_node) -> void {
  CARBON_DIAGNOSTIC(ModifierRepeated, Error, "`{0}` repeated on declaration",
                    Lex::TokenKind);
  context.emitter()
      .Build(second_node, ModifierRepeated, context.token_kind(second_node))
      .Note(first_node, ModifierPrevious, context.token_kind(first_node))
      .Emit();
}

static auto DiagnoseNotAllowedWith(Context& context, Parse::NodeId first_node,
                                   Parse::NodeId second_node) -> void {
  CARBON_DIAGNOSTIC(ModifierNotAllowedWith, Error,
                    "`{0}` not allowed on declaration with `{1}`",
                    Lex::TokenKind, Lex::TokenKind);
  context.emitter()
      .Build(second_node, ModifierNotAllowedWith,
             context.token_kind(second_node), context.token_kind(first_node))
      .Note(first_node, ModifierPrevious, context.token_kind(first_node))
      .Emit();
}

// Handles the keyword that starts a modifier. This may a standalone keyword,
// such as `private`, or the first in a complex modifier, such as `extern` in
// `extern library ...`. If valid, adds it to the modifier set and returns true.
// Otherwise, diagnoses and returns false.
static auto HandleModifier(Context& context, Parse::NodeId node_id,
                           KeywordModifierSet keyword) -> bool {
  auto& s = context.decl_introducer_state_stack().innermost();

  ModifierOrder order;
  KeywordModifierSet later_modifiers;
  if (keyword.HasAnyOf(KeywordModifierSet::Access)) {
    order = ModifierOrder::Access;
    later_modifiers = KeywordModifierSet::Extern | KeywordModifierSet::Decl;
  } else if (keyword.HasAnyOf(KeywordModifierSet::Extern)) {
    order = ModifierOrder::Extern;
    later_modifiers = KeywordModifierSet::Decl;
  } else {
    order = ModifierOrder::Decl;
    later_modifiers = KeywordModifierSet::None;
  }

  auto current_modifier_node_id = s.modifier_node_id(order);
  if (s.modifier_set.HasAnyOf(keyword)) {
    DiagnoseRepeated(context, current_modifier_node_id, node_id);
    return false;
  }
  if (current_modifier_node_id.has_value()) {
    DiagnoseNotAllowedWith(context, current_modifier_node_id, node_id);
    return false;
  }
  if (auto later_modifier_set = s.modifier_set & later_modifiers;
      !later_modifier_set.empty()) {
    // At least one later modifier is present. Diagnose using the closest.
    Parse::NodeId closest_later_modifier = Parse::NodeId::None;
    for (auto later_order = static_cast<int8_t>(order) + 1;
         later_order <= static_cast<int8_t>(ModifierOrder::Last);
         ++later_order) {
      if (s.ordered_modifier_node_ids[later_order].has_value()) {
        closest_later_modifier = s.ordered_modifier_node_ids[later_order];
        break;
      }
    }
    CARBON_CHECK(closest_later_modifier.has_value());

    CARBON_DIAGNOSTIC(ModifierMustAppearBefore, Error,
                      "`{0}` must appear before `{1}`", Lex::TokenKind,
                      Lex::TokenKind);
    context.emitter()
        .Build(node_id, ModifierMustAppearBefore, context.token_kind(node_id),
               context.token_kind(closest_later_modifier))
        .Note(closest_later_modifier, ModifierPrevious,
              context.token_kind(closest_later_modifier))
        .Emit();
    return false;
  }

  s.modifier_set.Add(keyword);
  s.set_modifier_node_id(order, node_id);
  return true;
}

#define CARBON_PARSE_NODE_KIND(Name)
#define CARBON_PARSE_NODE_KIND_TOKEN_MODIFIER(Name)                       \
  auto HandleParseNode(Context& context, Parse::Name##ModifierId node_id) \
      -> bool {                                                           \
    HandleModifier(context, node_id, KeywordModifierSet::Name);           \
    return true;                                                          \
  }
#include "toolchain/parse/node_kind.def"

auto HandleParseNode(Context& context,
                     Parse::ExternModifierWithLibraryId node_id) -> bool {
  auto name_literal_id = context.node_stack().Pop<SemIR::LibraryNameId>();
  if (HandleModifier(context, node_id, KeywordModifierSet::Extern)) {
    auto& s = context.decl_introducer_state_stack().innermost();
    s.extern_library = name_literal_id;
  }
  return true;
}

auto HandleParseNode(Context& context, Parse::ExternModifierId node_id)
    -> bool {
  return HandleModifier(context, node_id, KeywordModifierSet::Extern);
}

auto HandleParseNode(Context& context, Parse::ReturnedModifierId node_id)
    -> bool {
  return HandleModifier(context, node_id, KeywordModifierSet::Returned);
}

}  // namespace Carbon::Check
