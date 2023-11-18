// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/handle_modifier.h"

#include "toolchain/check/context.h"
#include "toolchain/lex/token_kind.h"

namespace Carbon::Check {

auto HandleDeclModifierKeyword(Context& context, Parse::Node parse_node)
    -> bool {
  context.node_stack().Push(parse_node);
  return true;
}

auto ValidateModifiers(Context& context, DeclModifierKeywords allowed,
                       std::function<Parse::Node()> pop_introducer)
    -> std::pair<DeclModifierKeywords, Parse::Node> {
  DeclModifierKeywords found;
  Parse::Node saw_access = Parse::Node::Invalid;
  Parse::Node saw_other = Parse::Node::Invalid;
  llvm::SmallVector<Parse::Node> modifier_nodes;
  // Note that since we are reading the modifier keywords off of a stack,
  // we get them in reverse order compared to how they appeared in the source.
  while (
      auto modifier_node =
          context.node_stack()
              .PopForSoloParseNodeIf<Parse::NodeKind::DeclModifierKeyword>()) {
    modifier_nodes.push_back(*modifier_node);
  }
  // FIXME: maybe pop this and return it?
  Parse::Node introducer_node = pop_introducer();
  Parse::Node returned_node =
      modifier_nodes.empty() ? introducer_node : modifier_nodes.back();

  // Reverse the order of processing the modifier nodes back to source order.
  while (!modifier_nodes.empty()) {
    auto modifier_node = modifier_nodes.back();
    modifier_nodes.pop_back();
    auto modifier_token = context.parse_tree().node_token(modifier_node);

    CARBON_DIAGNOSTIC(ModifierNotAllowedOnDeclaration, Error,
                      "Modifier not allowed.");
    CARBON_DIAGNOSTIC(ModifierDeclarationNote, Note, "On this declaration.");
    CARBON_DIAGNOSTIC(ModifierDuplicated, Error,
                      "Modifier repeated on the same declaration.");
    CARBON_DIAGNOSTIC(ModifierDuplicatedPrevious, Note,
                      "Previously appeared here.");
    CARBON_DIAGNOSTIC(ModifierNotAllowedWith, Error,
                      "Modifier not allowed on the same declaration.");
    CARBON_DIAGNOSTIC(ModifierNotAllowedWithPrevious, Note,
                      "With this modifier.");
    CARBON_DIAGNOSTIC(ModifierInWrongOrderSecond, Error,
                      "Modifier must appear earlier.");
    CARBON_DIAGNOSTIC(ModifierInWrongOrderFirst, Note, "Before this modifier.");

    switch (context.tokens().GetKind(modifier_token)) {
#define ACCESS(name)                                             \
  {                                                              \
    if (!allowed.name) {                                         \
      context.emitter()                                          \
          .Build(modifier_node, ModifierNotAllowedOnDeclaration) \
          .Note(introducer_node, ModifierDeclarationNote)        \
          .Emit();                                               \
    } else if (found.name) {                                     \
      context.emitter()                                          \
          .Build(modifier_node, ModifierDuplicated)              \
          .Note(saw_access, ModifierDuplicatedPrevious)          \
          .Emit();                                               \
    } else if (saw_access != Parse::Node::Invalid) {             \
      context.emitter()                                          \
          .Build(modifier_node, ModifierNotAllowedWith)          \
          .Note(saw_access, ModifierNotAllowedWithPrevious)      \
          .Emit();                                               \
    } else if (saw_other != Parse::Node::Invalid) {              \
      context.emitter()                                          \
          .Build(modifier_node, ModifierInWrongOrderSecond)      \
          .Note(saw_other, ModifierInWrongOrderFirst)            \
          .Emit();                                               \
    } else {                                                     \
      found.name = true;                                         \
      saw_access = modifier_node;                                \
    }                                                            \
    break;                                                       \
  }

      case Lex::TokenKind::Private:
        ACCESS(private_)
      case Lex::TokenKind::Protected:
        ACCESS(protected_)

#undef ACCESS
#define OTHER(name)                                              \
  {                                                              \
    if (!allowed.name) {                                         \
      context.emitter()                                          \
          .Build(modifier_node, ModifierNotAllowedOnDeclaration) \
          .Note(introducer_node, ModifierDeclarationNote)        \
          .Emit();                                               \
    } else if (found.name) {                                     \
      context.emitter()                                          \
          .Build(modifier_node, ModifierDuplicated)              \
          .Note(saw_other, ModifierDuplicatedPrevious)           \
          .Emit();                                               \
    } else if (saw_other != Parse::Node::Invalid) {              \
      context.emitter()                                          \
          .Build(modifier_node, ModifierNotAllowedWith)          \
          .Note(saw_other, ModifierNotAllowedWithPrevious)       \
          .Emit();                                               \
    } else {                                                     \
      found.name = true;                                         \
      saw_other = modifier_node;                                 \
    }                                                            \
    break;                                                       \
  }

      case Lex::TokenKind::Abstract:
        OTHER(abstract_)
      case Lex::TokenKind::Base:
        OTHER(base_)
      case Lex::TokenKind::Default:
        OTHER(default_)
      case Lex::TokenKind::Final:
        OTHER(final_)
      case Lex::TokenKind::Override:
        OTHER(override_)
      case Lex::TokenKind::Virtual:
        OTHER(virtual_)

#undef OTHER

      default: {
        CARBON_FATAL() << "Unhandled declaration modifier keyword";
        break;
      }
    }
  }
  return {found, returned_node};
}

}  // namespace Carbon::Check
