// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_NODE_KIND_H_
#define CARBON_TOOLCHAIN_PARSE_NODE_KIND_H_

#include <cstdint>

#include "common/enum_base.h"

namespace Carbon::Parse {

CARBON_DEFINE_RAW_ENUM_CLASS(NodeKind, uint8_t) {
#define CARBON_PARSE_NODE_KIND(Name) CARBON_RAW_ENUM_ENUMERATOR(Name)
#include "toolchain/parse/node_kind.def"
};

// A class wrapping an enumeration of the different kinds of nodes in the parse
// tree.
class NodeKind : public CARBON_ENUM_BASE(NodeKind) {
 public:
#define CARBON_PARSE_NODE_KIND(Name) CARBON_ENUM_CONSTANT_DECLARATION(Name)
#include "toolchain/parse/node_kind.def"

  // Returns true if the node is bracketed; otherwise, child_count is used.
  auto has_bracket() const -> bool;

  // Returns the bracketing node kind for the current node kind. Requires that
  // has_bracket is true.
  auto bracket() const -> NodeKind;

  // Returns the number of children that the node must have, often 0. Requires
  // that has_bracket is false.
  auto child_count() const -> int32_t;

  using EnumBase::Create;
};

#define CARBON_PARSE_NODE_KIND(Name) \
  CARBON_ENUM_CONSTANT_DEFINITION(NodeKind, Name)
#include "toolchain/parse/node_kind.def"

// We expect the parse node kind to fit compactly into 8 bits.
static_assert(sizeof(NodeKind) == 1, "Kind objects include padding!");

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_NODE_KIND_H_
