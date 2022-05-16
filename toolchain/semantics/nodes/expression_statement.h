// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_EXPRESSION_STATEMENT_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_EXPRESSION_STATEMENT_H_

#include "common/ostream.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/nodes/meta_node.h"

namespace Carbon::Semantics {

// Represents a statement that is only an expression, such as `Call()`.
class ExpressionStatement {
 public:
  static constexpr StatementKind MetaNodeKind =
      StatementKind::ExpressionStatement;

  explicit ExpressionStatement(Expression expression)
      : expression_(expression) {}

  auto expression() const -> Expression { return expression_; }

 private:
  Expression expression_;
};

}  // namespace Carbon::Semantics

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_EXPRESSION_STATEMENT_H_
