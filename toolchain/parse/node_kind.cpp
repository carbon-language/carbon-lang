// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/node_kind.h"

#include "common/check.h"

namespace Carbon::Parse {

CARBON_DEFINE_ENUM_CLASS_NAMES(NodeKind) = {
#define CARBON_PARSE_NODE_KIND(Name) CARBON_ENUM_CLASS_NAME_STRING(Name)
#include "toolchain/parse/node_kind.def"
};

auto NodeKind::has_bracket() const -> bool {
  static constexpr bool HasBracket[] = {
#define CARBON_PARSE_NODE_KIND_BRACKET(...) true,
#define CARBON_PARSE_NODE_KIND_CHILD_COUNT(...) false,
#include "toolchain/parse/node_kind.def"
  };
  return HasBracket[AsInt()];
}

auto NodeKind::bracket() const -> NodeKind {
  // Nodes are never self-bracketed, so we use that for nodes that instead set
  // child_count.
  static constexpr NodeKind Bracket[] = {
#define CARBON_PARSE_NODE_KIND_BRACKET(Name, BracketName, ...) \
  NodeKind::BracketName,
#define CARBON_PARSE_NODE_KIND_CHILD_COUNT(Name, ...) NodeKind::Name,
#include "toolchain/parse/node_kind.def"
  };
  auto bracket = Bracket[AsInt()];
  CARBON_CHECK(bracket != *this) << *this;
  return bracket;
}

auto NodeKind::child_count() const -> int32_t {
  static constexpr int32_t ChildCount[] = {
#define CARBON_PARSE_NODE_KIND_BRACKET(...) -1,
#define CARBON_PARSE_NODE_KIND_CHILD_COUNT(Name, Size, ...) Size,
#include "toolchain/parse/node_kind.def"
  };
  auto child_count = ChildCount[AsInt()];
  CARBON_CHECK(child_count >= 0) << *this;
  return child_count;
}

void CheckNodeMatchesLexerToken(NodeKind node_kind, Lex::TokenKind token_kind,
                                bool has_error) {
  switch (node_kind) {
#define CARBON_ANY_TOKEN return;

#define CARBON_TOKEN(Expected)                  \
  if (token_kind == Lex::TokenKind::Expected) { \
    return;                                     \
  }                                             \
  break;

#define CARBON_TOKEN_EITHER(Expected1, Expected2)               \
  if (token_kind == Lex::TokenKind::Expected1 ||                \
      (has_error && token_kind == Lex::TokenKind::Expected2)) { \
    return;                                                     \
  }                                                             \
  break;

#define CARBON_TOKEN_UNLESS_ERROR(Expected)                  \
  if (has_error || token_kind == Lex::TokenKind::Expected) { \
    return;                                                  \
  }                                                          \
  break;

#define CARBON_CASE(Name, MatchAction) \
  case NodeKind::Name:                 \
    MatchAction

#define CARBON_PARSE_NODE_KIND_BRACKET(Name, BracketName, MatchAction) \
  CARBON_CASE(Name, MatchAction)

#define CARBON_PARSE_NODE_KIND_CHILD_COUNT(Name, Size, MatchAction) \
  CARBON_CASE(Name, MatchAction)

#include "toolchain/parse/node_kind.def"
  }
  CARBON_FATAL() << "Created parse node with NodeKind " << node_kind
                 << " and has_error " << has_error
                 << " for unexpected lexical token " << token_kind;
}

}  // namespace Carbon::Parse
