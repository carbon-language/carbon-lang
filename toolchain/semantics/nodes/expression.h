// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_EXPRESSION_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_EXPRESSION_H_

#include "common/ostream.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/nodes/literal.h"

namespace Carbon::Semantics {

// Semantic information for an expression.
// TODO: This should probably be a MetaNode.
class Expression {
 public:
  explicit Expression(ParseTree::Node node, Literal literal)
      : node_(node), literal_(literal) {}

  auto node() const -> ParseTree::Node { return node_; }
  auto literal() const -> const Literal& { return literal_; }

 private:
  ParseTree::Node node_;
  Literal literal_;
};

}  // namespace Carbon::Semantics

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_EXPRESSION_H_
