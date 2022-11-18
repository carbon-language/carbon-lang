// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parse_node_kind.h"

#include "common/check.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon {

auto ParseNodeKind::name() const -> llvm::StringRef {
  static constexpr llvm::StringLiteral Names[] = {
#define CARBON_PARSE_NODE_KIND(Name) #Name,
#include "toolchain/parser/parse_node_kind.def"
  };
  return Names[static_cast<int>(kind_)];
}

auto ParseNodeKind::has_bracket() const -> bool {
  static constexpr bool HasBracket[] = {
#define CARBON_PARSE_NODE_KIND_BRACKET(...) true,
#define CARBON_PARSE_NODE_KIND_CHILD_COUNT(...) false,
#include "toolchain/parser/parse_node_kind.def"
  };
  return HasBracket[static_cast<int>(kind_)];
}

auto ParseNodeKind::bracket() const -> ParseNodeKind {
  // Nodes are never self-bracketed, so we use that for nodes that instead set
  // child_count.
  static constexpr ParseNodeKind Bracket[] = {
#define CARBON_PARSE_NODE_KIND_BRACKET(Name, BracketName) \
  ParseNodeKind::BracketName(),
#define CARBON_PARSE_NODE_KIND_CHILD_COUNT(Name, ...) ParseNodeKind::Name(),
#include "toolchain/parser/parse_node_kind.def"
  };
  auto bracket = Bracket[static_cast<int>(kind_)];
  CARBON_CHECK(bracket != kind_);
  return bracket;
}

auto ParseNodeKind::child_count() const -> int32_t {
  static constexpr int32_t ChildCount[] = {
#define CARBON_PARSE_NODE_KIND_BRACKET(...) -1,
#define CARBON_PARSE_NODE_KIND_CHILD_COUNT(Name, Size) Size,
#include "toolchain/parser/parse_node_kind.def"
  };
  auto child_count = ChildCount[static_cast<int>(kind_)];
  // TODO: Allows TodoFixParseNode; see parse_node_kind.def.
  CARBON_CHECK(child_count != -1);
  return child_count;
}

}  // namespace Carbon
