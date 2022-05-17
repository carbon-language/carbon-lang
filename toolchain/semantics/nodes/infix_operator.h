// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_INFIX_OPERATOR_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_INFIX_OPERATOR_H_

#include "common/ostream.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/nodes/meta_node.h"

namespace Carbon::Semantics {

// Represents an infix operator, such as `+` in `1 + 2`.
class InfixOperator {
 public:
  static constexpr ExpressionKind MetaNodeKind = ExpressionKind::InfixOperator;

  explicit InfixOperator(ParseTree::Node node, Expression lhs, Expression rhs)
      : node_(node), lhs_(lhs), rhs_(rhs) {}

  auto node() const -> ParseTree::Node { return node_; }
  auto lhs() const -> Expression { return lhs_; }
  auto rhs() const -> Expression { return rhs_; }

 private:
  ParseTree::Node node_;
  Expression lhs_;
  Expression rhs_;
};

}  // namespace Carbon::Semantics

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_INFIX_OPERATOR_H_
