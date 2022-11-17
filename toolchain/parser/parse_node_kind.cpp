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

auto ParseNodeKind::has_bracket() -> bool {
  static constexpr bool HasBracket[] = {
#define CARBON_PARSE_NODE_KIND_BRACKETED(...) true,
#define CARBON_PARSE_NODE_KIND_SIZED(...) false,
#include "toolchain/parser/parse_node_kind.def"
  };
  return HasBracket[static_cast<int>(kind_)];
}

auto ParseNodeKind::bracket() -> ParseNodeKind {
  static constexpr ParseNodeKind Bracket[] = {
#define CARBON_PARSE_NODE_KIND_BRACKETED(Name, BracketName) \
  ParseNodeKind::BracketName(),
#define CARBON_PARSE_NODE_KIND_SIZED(Name, ...) ParseNodeKind::Name(),
#include "toolchain/parser/parse_node_kind.def"
  };
  auto bracket = Bracket[static_cast<int>(kind_)];
  CARBON_CHECK(bracket != kind_);
  return bracket;
}

auto ParseNodeKind::subtree_size() -> int32_t {
  static constexpr int32_t SubtreeSize[] = {
#define CARBON_PARSE_NODE_KIND_BRACKETED() -1,
#define CARBON_PARSE_NODE_KIND_SIZED(Name, Size) Size,
#include "toolchain/parser/parse_node_kind.def"
  };
  auto subtree_size = SubtreeSize[static_cast<int>(kind_)];
  CARBON_CHECK(subtree_size != -1);
  return subtree_size;
}

}  // namespace Carbon
