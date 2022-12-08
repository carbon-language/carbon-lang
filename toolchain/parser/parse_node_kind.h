// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSER_PARSE_NODE_KIND_H_
#define CARBON_TOOLCHAIN_PARSER_PARSE_NODE_KIND_H_

#include <cstdint>

#include "common/ostream.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon {

#define CARBON_ENUM_BASE_NAME ParseNodeKindBase
#define CARBON_ENUM_DEF_PATH "toolchain/parser/parse_node_kind.def"
#include "toolchain/common/enum_base.def"

// A class wrapping an enumeration of the different kinds of nodes in the parse
// tree.
class ParseNodeKind : public ParseNodeKindBase<ParseNodeKind> {
 public:
  // Returns true if the node is bracketed; otherwise, child_count is used.
  auto has_bracket() const -> bool;

  // Returns the bracketing node kind for the current node kind. Requires that
  // has_bracket is true.
  auto bracket() const -> ParseNodeKind;

  // Returns the number of children that the node must have, often 0. Requires
  // that has_bracket is false.
  auto child_count() const -> int32_t;

 private:
  using ParseNodeKindBase::ParseNodeKindBase;
};

// We expect the parse node kind to fit compactly into 8 bits.
static_assert(sizeof(ParseNodeKind) == 1, "Kind objects include padding!");

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_PARSER_PARSE_NODE_KIND_H_
