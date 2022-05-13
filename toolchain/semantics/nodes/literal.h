// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_LITERAL_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_LITERAL_H_

#include "common/ostream.h"
#include "toolchain/parser/parse_tree.h"

namespace Carbon::Semantics {

// Represents all kinds of literals: `1`, `i32`, etc.
class Literal {
 public:
  explicit Literal(ParseTree::Node node) : node_(node) {}

  auto node() const -> ParseTree::Node { return node_; }

 private:
  ParseTree::Node node_;
};

}  // namespace Carbon::Semantics

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_LITERAL_H_
