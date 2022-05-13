// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_RETURN_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_RETURN_H_

#include "common/ostream.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/nodes/meta_node.h"

namespace Carbon::Semantics {

// Represents `return [expr];`
class Return {
 public:
  static constexpr StatementKind MetaNodeKind = StatementKind::Return;

  Return(ParseTree::Node node, llvm::Optional<Expression> expr)
      : node_(node), expr_(expr) {}

  auto node() const -> ParseTree::Node { return node_; }
  auto expr() const -> const llvm::Optional<Expression>& { return expr_; }

 private:
  ParseTree::Node node_;
  llvm::Optional<Expression> expr_;
};

}  // namespace Carbon::Semantics

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_RETURN_H_
