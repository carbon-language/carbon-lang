// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/node_kind.h"

#include "common/check.h"
#include "llvm/ADT/StringExtras.h"
#include "toolchain/parse/typed_nodes.h"

namespace Carbon::Parse {

auto operator<<(llvm::raw_ostream& output, NodeCategory category)
    -> llvm::raw_ostream& {
  if (!category) {
    output << "<none>";
  } else {
    llvm::ListSeparator sep("|");

#define CARBON_NODE_CATEGORY(Name)         \
  if (!!(category & NodeCategory::Name)) { \
    output << sep << #Name;                \
  }
    CARBON_NODE_CATEGORY(Decl);
    CARBON_NODE_CATEGORY(Expr);
    CARBON_NODE_CATEGORY(MemberName);
    CARBON_NODE_CATEGORY(Modifier);
    CARBON_NODE_CATEGORY(NameComponent);
    CARBON_NODE_CATEGORY(Pattern);
    CARBON_NODE_CATEGORY(Statement);
#undef CARBON_NODE_CATEGORY
  }
  return output;
}

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

auto NodeKind::CheckMatchesTokenKind(Lex::TokenKind token_kind, bool has_error)
    -> void {
  static constexpr Lex::TokenKind TokenIfValid[] = {
#define CARBON_IF_VALID(LexTokenKind) LexTokenKind
#define CARBON_PARSE_NODE_KIND_BRACKET(Name, BracketName, LexTokenKind) \
  Lex::TokenKind::LexTokenKind,
#define CARBON_PARSE_NODE_KIND_CHILD_COUNT(Name, Size, LexTokenKind) \
  Lex::TokenKind::LexTokenKind,
#include "toolchain/parse/node_kind.def"
  };
  static constexpr Lex::TokenKind TokenIfError[] = {
#define CARBON_IF_VALID(LexTokenKind) Error
#define CARBON_PARSE_NODE_KIND_BRACKET(Name, BracketName, LexTokenKind) \
  Lex::TokenKind::LexTokenKind,
#define CARBON_PARSE_NODE_KIND_CHILD_COUNT(Name, Size, LexTokenKind) \
  Lex::TokenKind::LexTokenKind,
#include "toolchain/parse/node_kind.def"
  };

  Lex::TokenKind expected_token_kind =
      has_error ? TokenIfError[AsInt()] : TokenIfValid[AsInt()];
  // Error indicates that the kind shouldn't be enforced.
  CARBON_CHECK(Lex::TokenKind::Error == expected_token_kind ||
               token_kind == expected_token_kind)
      << "Created parse node with NodeKind " << *this << " and has_error "
      << has_error << " for lexical token kind " << token_kind
      << ", but expected token kind " << expected_token_kind;
}

auto NodeKind::category() const -> NodeCategory {
  return definition().category();
}

auto NodeKind::definition() const -> const Definition& {
  static constexpr const Definition* Table[] = {
#define CARBON_PARSE_NODE_KIND(Name) &Parse::Name::Kind,
#include "toolchain/parse/node_kind.def"
  };
  return *Table[AsInt()];
}

}  // namespace Carbon::Parse
