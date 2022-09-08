// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_BINARY_OPERATOR_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_BINARY_OPERATOR_H_

#include "common/ostream.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/node_kind.h"

namespace Carbon::Semantics {

// Represents a binary operator, such as `+` in `1 + 2`.
class BinaryOperator {
 public:
  enum class Op {
    Add,
  };

  static constexpr NodeKind Kind = NodeKind::BinaryOperator;

  explicit BinaryOperator(ParseTree::Node node, NodeId id, Op op, NodeId lhs_id,
                          NodeId rhs_id)
      : node_(node), id_(id), op_(op), lhs_id_(lhs_id), rhs_id_(rhs_id) {}

  void Print(llvm::raw_ostream& out) const {
    out << "BinaryOperator(" << id_ << ", ";
    switch (op_) {
      case Op::Add:
        out << "+";
        break;
    }
    out << ", " << lhs_id_ << ", %" << rhs_id_ << ")";
  }

  auto node() const -> ParseTree::Node { return node_; }
  auto id() const -> NodeId { return id_; }
  auto op() const -> Op { return op_; }
  auto lhs_id() const -> NodeId { return lhs_id_; }
  auto rhs_id() const -> NodeId { return rhs_id_; }

 private:
  ParseTree::Node node_;
  NodeId id_;
  Op op_;
  NodeId lhs_id_;
  NodeId rhs_id_;
};

}  // namespace Carbon::Semantics

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_BINARY_OPERATOR_H_
