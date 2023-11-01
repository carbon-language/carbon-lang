// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/node_kind.h"

#include "common/check.h"

namespace Carbon::Parse {

CARBON_DEFINE_ENUM_CLASS_NAMES(LampKind) = {
#define CARBON_PARSE_NODE_KIND(Name) CARBON_ENUM_CLASS_NAME_STRING(Name)
#include "toolchain/parse/node_kind.def"
};

auto LampKind::has_bracket() const -> bool {
  static constexpr bool HasBracket[] = {
#define CARBON_PARSE_NODE_KIND_BRACKET(...) true,
#define CARBON_PARSE_NODE_KIND_CHILD_COUNT(...) false,
#include "toolchain/parse/node_kind.def"
  };
  return HasBracket[AsInt()];
}

auto LampKind::bracket() const -> LampKind {
  // Lamps are never self-bracketed, so we use that for nodes that instead set
  // child_count.
  static constexpr LampKind Bracket[] = {
#define CARBON_PARSE_NODE_KIND_BRACKET(Name, BracketName, ...) \
  LampKind::BracketName,
#define CARBON_PARSE_NODE_KIND_CHILD_COUNT(Name, ...) LampKind::Name,
#include "toolchain/parse/node_kind.def"
  };
  auto bracket = Bracket[AsInt()];
  CARBON_CHECK(bracket != *this) << *this;
  return bracket;
}

auto LampKind::child_count() const -> int32_t {
  static constexpr int32_t ChildCount[] = {
#define CARBON_PARSE_NODE_KIND_BRACKET(...) -1,
#define CARBON_PARSE_NODE_KIND_CHILD_COUNT(Name, Size, ...) Size,
#include "toolchain/parse/node_kind.def"
  };
  auto child_count = ChildCount[AsInt()];
  CARBON_CHECK(child_count >= 0) << *this;
  return child_count;
}

void CheckNodeMatchesLexerToken(LampKind node_kind, Lex::TokenKind token_kind,
                                bool has_error) {
  switch (node_kind) {
    // Use `CARBON_LOG CARBON_ANY_TOKEN` to discover which combinations happen
    // in practice.
#define CARBON_LOG                                                        \
  llvm::errs() << "ZZZ: Created parse node with LampKind " << node_kind   \
               << " and has_error " << has_error << " for lexical token " \
               << token_kind << "\n";

#define CARBON_ANY_TOKEN return;

#define CARBON_TOKEN(Expected)                  \
  if (token_kind == Lex::TokenKind::Expected) { \
    return;                                     \
  }

#define CARBON_IF_ERROR(MatchActions) \
  if (has_error) {                    \
    MatchActions                      \
  }

#define CARBON_CASE(Name, MatchActions) \
  case LampKind::Name:                  \
    MatchActions;                       \
    break;

#define CARBON_PARSE_NODE_KIND_BRACKET(Name, BracketName, MatchActions) \
  CARBON_CASE(Name, MatchActions)

#define CARBON_PARSE_NODE_KIND_CHILD_COUNT(Name, Size, MatchActions) \
  CARBON_CASE(Name, MatchActions)

#include "toolchain/parse/node_kind.def"

#undef CARBON_LOG
#undef CARBON_CASE
  }
  CARBON_FATAL() << "Created parse node with LampKind " << node_kind
                 << " and has_error " << has_error
                 << " for unexpected lexical token " << token_kind;
}

}  // namespace Carbon::Parse
