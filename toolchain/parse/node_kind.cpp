// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/node_kind.h"

#include "llvm/ADT/StringExtras.h"
#include "toolchain/parse/typed_nodes.h"

namespace Carbon::Parse {

CARBON_DEFINE_ENUM_CLASS_NAMES(NodeKind) = {
#define CARBON_PARSE_NODE_KIND(Name) CARBON_ENUM_CLASS_NAME_STRING(Name)
#include "toolchain/parse/node_kind.def"
};

// Check that each typed node defines a `token` member.
#define CARBON_PARSE_NODE_KIND(Name) \
  static_assert(requires(Name node) { node.token; });
#include "toolchain/parse/node_kind.def"

auto NodeKind::has_bracket() const -> bool {
  return definition().has_bracket();
}

auto NodeKind::bracket() const -> NodeKind { return definition().bracket(); }

auto NodeKind::has_child_count() const -> bool {
  return definition().has_child_count();
}

auto NodeKind::child_count() const -> int32_t {
  return definition().child_count();
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
