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

auto ValidateModifiers(Context& context, DeclModifierKeywords allowed)
    -> DeclModifierKeywords {
  DeclModifierKeywords found;
  std::optional<Parse::Node> saw_access;
  std::optional<Parse::Node> saw_other;

  while (
      auto modifier_node =
          context.node_stack()
              .PopForSoloParseNodeIf<Parse::NodeKind::DeclModifierKeyword>()) {
    auto modifier_token = context.parse_tree().node_token(*modifier_node);
    switch (context.tokens().GetKind(modifier_token)) {
#define ACCESS(name)             \
  {                              \
    if (!allowed.name) {         \
      CARBON_FATAL() << "FIXME"; \
    } else if (found.name) {     \
      CARBON_FATAL() << "FIXME"; \
    } else if (saw_other) {      \
      CARBON_FATAL() << "FIXME"; \
    } else if (saw_access) {     \
      CARBON_FATAL() << "FIXME"; \
    }                            \
    found.name = true;           \
    saw_access = modifier_node;  \
    break;                       \
  }

      case Lex::TokenKind::Private:
        ACCESS(private_)
      case Lex::TokenKind::Protected:
        ACCESS(protected_)

#undef ACCESS
#define OTHER(name)              \
  {                              \
    if (!allowed.name) {         \
      CARBON_FATAL() << "FIXME"; \
    } else if (found.name) {     \
      CARBON_FATAL() << "FIXME"; \
    } else if (saw_other) {      \
      CARBON_FATAL() << "FIXME"; \
    }                            \
    found.name = true;           \
    saw_other = modifier_node;   \
    break;                       \
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
  return found;
}

}  // namespace Carbon::Check
